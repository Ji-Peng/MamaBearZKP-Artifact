use std::{collections::HashMap, marker::PhantomData};

use arithmetic::{
    field::{as_bytes_vec, FftField, Field},
    mul_group::Radix2Group,
    poly::MultiLinearPoly,
};
use util::{
    fiat_shamir::Proof,
    merkle_tree::{MerkleTreeProver, MerkleTreeVerifier, HASH_SIZE},
};

use crate::Transcript;

use super::{CommitmentSerde, PolyCommitProver, PolyCommitVerifier};

/// DeepFold DEEP (out-of-domain) power vector.
///
/// Given a base field element `base` and a length `len`, returns the vector
/// `[base^(2^0), base^(2^1), base^(2^2), ..., base^(2^(len-1))]` obtained by
/// repeated squaring.
///
/// This is the shape forced by the twin-polynomial relation used throughout
/// DeepFold: the univariate polynomial `f^(0)` and its `mu`-variate multilinear
/// twin `f~` satisfy `f^(0)(x) = f~(x, x^2, x^4, ..., x^(2^(mu-1)))`. Because our
/// multilinear-eval helper (`eval_multilinear_ext`) folds coordinate `point[0]`
/// first (the low-bit / round-0 variable), coordinate `k` of this vector must be
/// `base^(2^k)`. Evaluating the committed multilinear at such a power vector
/// therefore collapses to a single univariate evaluation `f^(0)(base)` at the
/// random field point `base` — exactly the out-of-domain evaluation DeepFold's
/// Theorem 4 (binding) and Lemma 7 (per-round codeword uniqueness) require.
///
/// After each folding round the leading coordinate is dropped, so at round `i`
/// the surviving vector is `[base^(2^i), ..., base^(2^(mu-1))]`.
///
/// Domain note: when `base` is a Montgomery-form field element (as in the packed
/// MamaBear paths), `*` performs Montgomery multiplication, so the result
/// is the Montgomery-form power vector — exactly what the multilinear-eval helpers
/// there expect. In the generic scalar path `base` is a normal-form element.
pub(crate) fn deep_power_vector<F: Field>(base: F, len: usize) -> Vec<F> {
    let mut v = Vec::with_capacity(len);
    let mut cur = base;
    for _ in 0..len {
        v.push(cur);
        cur = cur * cur;
    }
    v
}

#[derive(Debug, Clone, Default)]
pub struct MerkleRoot(pub [u8; HASH_SIZE]);

impl CommitmentSerde for MerkleRoot {
    fn size(_nv: usize, _np: usize) -> usize {
        HASH_SIZE
    }

    fn serialize_into(&self, buffer: &mut [u8]) {
        buffer.copy_from_slice(&self.0);
    }

    fn deserialize_from(proof: &mut Proof, _nv: usize, _np: usize) -> Self {
        let root = proof.get_next_hash();
        Self(root)
    }
}

#[derive(Debug, Clone)]
pub struct DeepFoldParam<F: FftField> {
    pub mult_subgroups: Vec<Radix2Group<F::FftBaseField>>,
    pub variable_num: usize,
    pub query_num: usize,
}

#[derive(Clone)]
pub struct QueryResult<F: Field> {
    pub proof_bytes: Vec<u8>,
    pub proof_values: HashMap<usize, F>,
}

impl<F: Field> QueryResult<F> {
    pub fn verify_merkle_tree(
        &self,
        leaf_indices: &Vec<usize>,
        leaf_size: usize,
        merkle_verifier: &MerkleTreeVerifier,
    ) -> bool {
        let len = merkle_verifier.leave_number;
        let leaves: Vec<Vec<u8>> = leaf_indices
            .iter()
            .map(|i| {
                as_bytes_vec(
                    &(0..leaf_size)
                        .map(|j| self.proof_values.get(&(i + j * len)).unwrap().clone())
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let res = merkle_verifier.verify(self.proof_bytes.clone(), leaf_indices, &leaves);
        assert!(res);
        res
    }
}

#[derive(Clone)]
pub struct InterpolateValue<F: Field> {
    pub value: Vec<F>,
    leaf_size: usize,
    merkle_tree: MerkleTreeProver,
}

impl<F: FftField> InterpolateValue<F> {
    pub fn new(value: Vec<F>, leaf_size: usize) -> Self {
        let len = value.len() / leaf_size;
        let merkle_tree = MerkleTreeProver::new(
            (0..len)
                .map(|i| {
                    as_bytes_vec::<F>(
                        &(0..leaf_size)
                            .map(|j| value[len * j + i])
                            .collect::<Vec<_>>(),
                    )
                })
                .collect(),
        );
        Self {
            value,
            leaf_size,
            merkle_tree,
        }
    }

    pub fn leave_num(&self) -> usize {
        self.merkle_tree.leave_num()
    }

    pub fn commit(&self) -> [u8; HASH_SIZE] {
        self.merkle_tree.commit()
    }

    pub fn query(&self, leaf_indices: &Vec<usize>) -> (Vec<u8>, Vec<F>) {
        let len = self.merkle_tree.leave_num();
        assert_eq!(len * self.leaf_size, self.value.len());
        let proof_values = (0..self.leaf_size)
            .flat_map(|i| {
                leaf_indices
                    .iter()
                    .map(|j| self.value[j.clone() + i * len])
                    .collect::<Vec<_>>()
            })
            .collect();
        let proof_bytes = self.merkle_tree.open(&leaf_indices);
        (proof_bytes, proof_values)
    }
}

#[derive(Clone)]
pub struct DeepFoldProver<F: FftField> {
    pub interpolation: InterpolateValue<F::FftBaseField>,
    poly: Vec<Vec<F::BaseField>>,
}

impl<F: FftField> DeepFoldProver<F> {
    /// Full DeepFold open, parameterized by `deep_c_offset` which is added to the OOD claim
    /// `c = f^(0)(alpha)`. In production the trait `open` passes `F::zero()` (honest c). The
    /// decoupled-c soundness PoC passes a nonzero offset to forge a transcript-consistent
    /// proof whose committed `c` does NOT match the polynomial, and checks the verifier's
    /// DEEP terminal check rejects it.
    pub(crate) fn open_inner(
        pp: &DeepFoldParam<F>,
        provers: Vec<&Self>,
        point: Vec<F>,
        transcript: &mut Transcript,
        deep_c_offset: F,
    ) {
        let mut interpolations: Vec<InterpolateValue<F>> = vec![];
        let r: F = transcript.challenge_f();
        let mut poly_evals = provers[0].poly[0]
            .iter()
            .map(|x| F::from(*x))
            .collect::<Vec<_>>();
        for i in 0..provers.len() {
            let start = if i == 0 { 1 } else { 0 };
            for j in start..provers[i].poly.len() {
                for k in 0..poly_evals.len() {
                    poly_evals[k] *= r;
                    poly_evals[k].add_assign_base_elem(provers[i].poly[j][k]);
                }
            }
        }
        let len = pp.mult_subgroups[0].size();
        let mut poly_interpolations = (0..len).map(|_| F::zero()).collect::<Vec<_>>();
        for i in 0..provers.len() {
            for j in 0..len {
                for k in 0..provers[i].poly.len() {
                    poly_interpolations[j] *= r;
                    poly_interpolations[j] += F::from(provers[i].interpolation.value[j + len * k]);
                }
            }
        }
        // === DeepFold DEEP (out-of-domain) binding — prover side ===
        //
        // A bare Merkle root `rt_0` does not bind the committed vector to a unique
        // polynomial in the list-decoding regime (several codewords may lie within the
        // decoding radius). DeepFold fixes this with an out-of-domain challenge `alpha`
        // and the extra evaluation `c = f^(0)(alpha)`: for a random `alpha`, two distinct
        // list-decoded codewords disagree at `alpha` with overwhelming probability
        // (Theorem 4). We keep `c` in the evaluation proof, NOT in the commitment struct:
        // `alpha` is Fiat-Shamir-derived after the caller absorbed `rt_0`, so the
        // committed vector is already fixed and binding holds regardless of when `alpha`
        // is drawn.
        let alpha: F = transcript.challenge_f();
        let alpha_vec = deep_power_vector(alpha, pp.variable_num);
        // c = f^(0)(alpha) = f~(alpha, alpha^2, ...), evaluated on the same combined
        // multilinear the z-line uses, so the DEEP claim rides the identical rail.
        // `deep_c_offset` is zero in production; the soundness PoC uses a nonzero value.
        transcript.append_f(MultiLinearPoly::eval_multilinear_ext(&poly_evals, &alpha_vec) + deep_c_offset);
        // The growing DEEP point set A. Entry 0 is the evaluation point z (the existing
        // "z-line"); entry 1 is the OOD point alpha. Each round we (a) add a fresh
        // per-round point alpha_i (Lemma 7: a random point that pins the codeword so a
        // malicious prover cannot swap codewords between rounds), and (b) for every active
        // point send one field element that, with its carried claim, determines the
        // round's degree-1 linear function. The prover only needs the point vectors; the
        // verifier reconstructs and checks the claims.
        let mut active: Vec<Vec<F>> = vec![point.clone(), alpha_vec];
        for i in 0..pp.variable_num {
            // (a) Introduce the fresh round-i DEEP point alpha_i (every round but the
            // last terminal round, where all chains converge to f^(mu)). Its seed is
            // f^(i)(alpha_i) = f~(r_0..r_{i-1}, alpha_i, alpha_i^2, ...) — the DEEP
            // evaluation the protocol adds each round. It has no carried claim yet, so
            // the seed is sent explicitly (the verifier cannot derive it).
            if i < pp.variable_num - 1 {
                let alpha_i: F = transcript.challenge_f();
                let w_i = deep_power_vector(alpha_i, pp.variable_num - i);
                transcript.append_f(MultiLinearPoly::eval_multilinear_ext(&poly_evals, &w_i));
                active.push(w_i);
            }
            // (b) For every active point (z first, then alpha, then the fresh alpha_j's),
            // send the line's value at (head+1). With the carried claim at `head` (which
            // the verifier already holds) this fixes g(X) = f~(r_0..r_{i-1}, X, tail);
            // using head+1 makes the interpolation denominator 1 (the original z-line
            // trick). The z-point's element here is byte-identical to the old z-line.
            for w in &active {
                let mut off = w.clone();
                off[0].add_assign_base_elem(F::BaseField::one());
                transcript.append_f(MultiLinearPoly::eval_multilinear_ext(&poly_evals, &off));
            }
            // Fold challenge r_i — drawn AFTER all round-i sends so the prover commits
            // every DEEP off-value before learning r_i (Fiat-Shamir / Schwartz-Zippel).
            let challenge: F = transcript.challenge_f();
            let new_len = poly_evals.len() / 2;
            for j in 0..new_len {
                poly_evals[j] =
                    poly_evals[j * 2] + (poly_evals[j * 2 + 1] - poly_evals[j * 2]) * challenge;
            }
            poly_evals.truncate(new_len);
            // Head-drop: variable i is now folded into r_i, so every active point loses
            // its leading coordinate. After this, active[0] == point[i+1..] (the z-line).
            for w in &mut active {
                w.remove(0);
            }
            let next_evaluation = Self::evaluate_next_domain(
                if i == 0 {
                    &poly_interpolations
                } else {
                    &interpolations[i - 1].value
                },
                pp,
                i,
                challenge,
            );
            if i < pp.variable_num - 1 {
                let new_interpolation = InterpolateValue::new(next_evaluation, 2);
                transcript.append_u8_slice(&new_interpolation.commit(), HASH_SIZE);
                interpolations.push(new_interpolation);
            } else {
                transcript.append_f(next_evaluation[0]);
            }
        }
        let mut leaf_indices = transcript.challenge_usizes(pp.query_num);
        for i in 0..pp.variable_num {
            let len = pp.mult_subgroups[i].size();
            leaf_indices = leaf_indices.iter_mut().map(|v| *v % (len >> 1)).collect();
            leaf_indices.sort();
            leaf_indices.dedup();
            if i == 0 {
                let query = provers
                    .iter()
                    .map(|j| j.interpolation.query(&leaf_indices))
                    .collect::<Vec<_>>();
                for q in query {
                    transcript.append_u8_slice(&q.0, q.0.len());
                    for j in q.1 {
                        transcript.append_f(j);
                    }
                }
            } else {
                let query = interpolations[i - 1].query(&leaf_indices);
                transcript.append_u8_slice(&query.0, query.0.len());
                for j in query.1 {
                    transcript.append_f(j);
                }
            }
        }
    }

    fn evaluate_next_domain(
        last_interpolation: &Vec<F>,
        pp: &DeepFoldParam<F>,
        round: usize,
        challenge: F,
    ) -> Vec<F> {
        let mut res = vec![];
        let len = pp.mult_subgroups[round].size();
        let subgroup = &pp.mult_subgroups[round];
        for i in 0..(len / 2) {
            let x = last_interpolation[i];
            let nx = last_interpolation[i + len / 2];
            let sum = x + nx;
            let new_v = sum + challenge * ((x - nx) * F::from(subgroup.element_inv_at(i)) - sum);
            res.push(new_v.mul_base_elem(<F as Field>::BaseField::inv_2()));
        }
        res
    }
}

impl<F: FftField> PolyCommitProver<F> for DeepFoldProver<F> {
    type Param = DeepFoldParam<F>;
    type Commitment = MerkleRoot;

    fn new(pp: &Self::Param, poly: &[Vec<F::BaseField>]) -> Self {
        let values = poly
            .iter()
            .flat_map(|x| pp.mult_subgroups[0].fft(x.clone()))
            .collect::<Vec<_>>();
        DeepFoldProver {
            interpolation: InterpolateValue::new(values, 2 * poly.len()),
            poly: poly.iter().map(|x| x.clone()).collect(),
        }
    }

    fn commit(&self) -> Self::Commitment {
        MerkleRoot(self.interpolation.commit())
    }

    fn open(pp: &Self::Param, provers: Vec<&Self>, point: Vec<F>, transcript: &mut Transcript) {
        // Production open: honest OOD claim (deep_c_offset = 0).
        Self::open_inner(pp, provers, point, transcript, F::zero());
    }
}

#[derive(Clone)]
pub struct DeepFoldVerifier<F: FftField> {
    commit: MerkleTreeVerifier,
    poly_num: usize,
    _data: PhantomData<F>,
}

impl<F: FftField> PolyCommitVerifier<F> for DeepFoldVerifier<F> {
    type Param = DeepFoldParam<F>;
    type Commitment = MerkleRoot;

    fn new(pp: &Self::Param, commit: Self::Commitment, poly_num: usize) -> Self {
        DeepFoldVerifier {
            commit: MerkleTreeVerifier::new(pp.mult_subgroups[0].size() / 2, commit.0),
            poly_num,
            _data: PhantomData::default(),
        }
    }

    fn verify(
        pp: &Self::Param,
        verifiers: Vec<&Self>,
        point: Vec<F>,
        evals: Vec<Vec<F>>,
        transcript: &mut Transcript,
        proof: &mut Proof,
    ) -> bool {
        let r = transcript.challenge_f();
        let mut eval = F::zero();
        for i in evals {
            for j in i {
                eval *= r;
                eval += j;
            }
        }
        // === DeepFold DEEP (out-of-domain) binding — verifier side (mirror of open) ===
        // Redraw the same alpha, read c = f^(0)(alpha), and seed the DEEP point set.
        // `eval` above is the combined claim y = f~(z), i.e. the z-point's carried claim;
        // `deep` carries the alpha-family points (the OOD alpha, then each per-round
        // alpha_i) as (current-vector, carried-claim). All chains must converge to the
        // same terminal f^(mu).
        let alpha: F = transcript.challenge_f();
        let alpha_vec = deep_power_vector(alpha, pp.variable_num);
        let c = proof.get_next_and_step::<F>();
        transcript.append_f(c);
        let mut deep: Vec<(Vec<F>, F)> = vec![(alpha_vec, c)];

        let mut challenges = vec![];
        let mut commits = vec![];
        for i in 0..point.len() {
            // (a) Fresh round-i point alpha_i: read its seed = f^(i)(alpha_i) and push it.
            if i < pp.variable_num - 1 {
                let alpha_i: F = transcript.challenge_f();
                let w_i = deep_power_vector(alpha_i, pp.variable_num - i);
                let seed = proof.get_next_and_step::<F>();
                transcript.append_f(seed);
                deep.push((w_i, seed));
            }
            // (b) Read the off-values in the exact order the prover sent them: z first,
            // then the alpha-family points (alpha, alpha_0, ..., alpha_i).
            let next_eval = proof.get_next_and_step::<F>();
            transcript.append_f(next_eval);
            let mut deep_offs = Vec::with_capacity(deep.len());
            for _ in 0..deep.len() {
                let off = proof.get_next_and_step::<F>();
                transcript.append_f(off);
                deep_offs.push(off);
            }
            // Fold challenge r_i — drawn after all reads, matching the prover.
            let challenge = transcript.challenge_f::<F>();

            // z-line claim update (head = point[i]); byte-identical to the old z-line.
            eval += (challenge - point[i]) * (next_eval - eval);
            // alpha-family claim updates: propagate each line to r_i, then head-drop.
            for (k, (w, claim)) in deep.iter_mut().enumerate() {
                let head = w[0];
                *claim = *claim + (challenge - head) * (deep_offs[k] - *claim);
                w.remove(0);
            }
            challenges.push(challenge);
            if i < pp.variable_num - 1 {
                let merkle_root = proof.get_next_hash();
                transcript.append_u8_slice(&merkle_root, HASH_SIZE);
                commits.push(MerkleTreeVerifier::new(
                    pp.mult_subgroups[i + 1].size() / 2,
                    merkle_root,
                ));
            } else {
                // Terminal round: every chain (z and all alpha-family points) must have
                // converged to the same value f^(mu) = f~(r_0..r_{mu-1}). The z-terminal
                // is `eval`; the prover's final scalar must equal it (existing check) and
                // — the DEEP binding — every alpha-family terminal claim must equal it too.
                // A prover that swapped codewords between rounds cannot satisfy all of
                // these, because the alpha_i were random and unknown at codeword-selection
                // time (Lemma 7).
                let final_value = proof.get_next_and_step::<F>();
                transcript.append_f(final_value);
                if final_value != eval {
                    return false;
                }
                for (_, claim) in &deep {
                    if *claim != eval {
                        return false;
                    }
                }
            }
        }

        let mut leaf_indices = transcript.challenge_usizes(pp.query_num);
        let mut indices = leaf_indices.clone();
        let mut query_results = vec![];
        for i in 0..pp.variable_num {
            let len = pp.mult_subgroups[i].size();
            leaf_indices = leaf_indices.iter_mut().map(|v| *v % (len >> 1)).collect();
            leaf_indices.sort();
            leaf_indices.dedup();

            if i == 0 {
                let mut poly_values = vec![];
                for j in 0..verifiers.len() {
                    let proof_bytes =
                        proof.get_next_slice(verifiers[j].commit.proof_length(&leaf_indices));
                    let proof_values = (0..leaf_indices.len() * 2 * verifiers[j].poly_num)
                        .map(|_| proof.get_next_and_step::<F::FftBaseField>())
                        .collect::<Vec<_>>();
                    transcript.append_u8_slice(&proof_bytes, proof_bytes.len());
                    for k in &proof_values {
                        transcript.append_f(*k);
                    }
                    poly_values.append(
                        &mut (0..verifiers[j].poly_num)
                            .map(|k| {
                                (&proof_values
                                    [k * leaf_indices.len() * 2..(k + 1) * leaf_indices.len() * 2])
                                    .to_vec()
                            })
                            .collect::<Vec<_>>(),
                    );
                    let query = QueryResult {
                        proof_bytes,
                        proof_values: proof_values
                            .into_iter()
                            .enumerate()
                            .map(|(idx, x)| {
                                (
                                    leaf_indices[idx % leaf_indices.len()]
                                        + (len / 2) * (idx / leaf_indices.len()),
                                    x,
                                )
                            })
                            .collect(),
                    };
                    assert!(query.verify_merkle_tree(
                        &leaf_indices,
                        2 * verifiers[j].poly_num,
                        &verifiers[j].commit
                    ));
                }
                let poly_values = (0..leaf_indices.len() * 2)
                    .into_iter()
                    .map(|j| {
                        let mut x = F::zero();
                        for k in 0..poly_values.len() {
                            x *= r;
                            x += F::from(poly_values[k][j]);
                        }
                        x
                    })
                    .collect::<Vec<_>>();

                query_results.push(QueryResult {
                    proof_bytes: vec![],
                    proof_values: leaf_indices
                        .iter()
                        .map(|&x| x)
                        .chain(leaf_indices.iter().map(|&x| x + len / 2))
                        .zip(poly_values)
                        .collect(),
                })
            } else {
                let proof_bytes = proof.get_next_slice(commits[i - 1].proof_length(&leaf_indices));
                let proof_values = (0..leaf_indices.len() * 2)
                    .map(|_| proof.get_next_and_step::<F>())
                    .collect::<Vec<_>>();
                transcript.append_u8_slice(&proof_bytes, proof_bytes.len());
                for j in &proof_values {
                    transcript.append_f(*j);
                }
                let query = QueryResult {
                    proof_bytes,
                    proof_values: leaf_indices
                        .iter()
                        .map(|&x| x)
                        .chain(leaf_indices.iter().map(|x| x + len / 2))
                        .zip(proof_values.into_iter())
                        .collect(),
                };
                query.verify_merkle_tree(&leaf_indices, 2, &commits[i - 1]);
                query_results.push(query);
            }
        }
        drop(leaf_indices);
        for i in 0..pp.variable_num {
            let len = pp.mult_subgroups[i].size();
            indices = indices.iter_mut().map(|v| *v % (len >> 1)).collect();
            indices.sort();
            indices.dedup();

            for j in indices.iter() {
                let x = query_results[i].proof_values.get(&j).unwrap().clone();
                let nx = query_results[i]
                    .proof_values
                    .get(&(j + len / 2))
                    .unwrap()
                    .clone();
                let sum = x + nx;
                let new_v = sum
                    + challenges[i]
                        * ((x - nx) * F::from(pp.mult_subgroups[i].element_inv_at(*j)) - sum);
                if i < pp.variable_num - 1 {
                    if new_v != query_results[i + 1].proof_values[j].double() {
                        println!("{} {}", file!(), line!());
                        return false;
                    }
                } else {
                    if new_v.mul_base_elem(<F as Field>::BaseField::inv_2()) != eval {
                        return false;
                    }
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod deep_ood_tests {
    use super::deep_power_vector;
    use arithmetic::field::{goldilocks64::Goldilocks64Ext, Field};
    use arithmetic::poly::MultiLinearPoly;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    type F = Goldilocks64Ext;

    // Oracle 1: deep_power_vector(alpha, n)[k] must be exactly alpha^(2^k). This is the
    // shape the twin relation forces; a round-trip test cannot catch a wrong-but-mutually
    // consistent power vector (prover and verifier would agree on the wrong shape), so we
    // pin it independently here.
    #[test]
    fn deep_power_vector_shape() {
        let mut rng = SmallRng::seed_from_u64(0xD00D_0001);
        for &n in &[1usize, 2, 3, 5, 8] {
            let alpha = F::random(&mut rng);
            let v = deep_power_vector(alpha, n);
            assert_eq!(v.len(), n);
            let mut expect = alpha;
            for k in 0..n {
                assert_eq!(v[k], expect, "deep_power_vector[{k}] != alpha^(2^{k})");
                expect = expect * expect;
            }
        }
    }

    // Oracle 2: the twin relation. For a multilinear f~ given by its monomial coefficients
    // c_j (indexed by the binary expansion of j), its hypercube values are
    // v_b = sum_{j subseteq b} c_j (the subset-sum / zeta transform). DeepFold's OOD claim
    // c = eval_multilinear_ext(v, deep_power_vector(alpha, mu)) must equal the univariate
    // twin f^(0)(alpha) = sum_j c_j * alpha^j. This is exactly the identity that makes the
    // OOD evaluation a genuine codeword evaluation at the random point alpha (Theorem 4).
    #[test]
    fn deep_ood_equals_univariate_twin() {
        let mut rng = SmallRng::seed_from_u64(0xD00D_0002);
        for &mu in &[1usize, 2, 3, 4] {
            let n = 1usize << mu;
            let coeffs: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
            // hypercube values from monomial coeffs
            let mut vals = vec![F::zero(); n];
            for b in 0..n {
                let mut s = F::zero();
                for j in 0..n {
                    if j & b == j {
                        s = s + coeffs[j];
                    }
                }
                vals[b] = s;
            }
            for _ in 0..8 {
                let alpha = F::random(&mut rng);
                let lhs =
                    MultiLinearPoly::eval_multilinear_ext(&vals, &deep_power_vector(alpha, mu));
                // univariate f^(0)(alpha) = sum_j coeffs[j] * alpha^j
                let mut rhs = F::zero();
                let mut pow = F::one();
                for j in 0..n {
                    rhs = rhs + coeffs[j] * pow;
                    pow = pow * alpha;
                }
                assert_eq!(lhs, rhs, "OOD eval != univariate twin at mu={mu}");
            }
        }
    }
}
