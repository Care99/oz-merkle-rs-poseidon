use ethers::abi::ethabi;
use ethers::core::utils::keccak256;
use ethers::types::{Address, H256, U256};
use ark_bls12_381::Fr as Frbls12_381;
pub struct MerkleTree {
    elements: Vec<H256>,
    layers: Vec<Vec<H256>>,
    leaves: usize,
}

pub mod read_constants;
use read_constants::*;

use ark_bls12_381::Fq as Fqbls12_381;
use ark_bls12_381::Fr as Frbls12_381;
use ark_bls12_381::Fq as Fqbls12_377;
use ark_bls12_377::Fr as Frbls12_377;
use ark_ff::BigInt as arkBigInt;
use ark_ff::PrimeField;

/*********************************************************
Hashing function
*********************************************************/
pub fn hash<T: PrimeField>(
    input: &Vec<T>,
    constants: &Constants<T>,
    output_length: u32,
    r: usize,
) -> Vec<T> {
    let mut state = absorb(input, constants, r);
    let output = squeeze(&mut state, constants, output_length, r);
    output.clone()
}

/*********************************************************
Squeezing stage
*********************************************************/
fn squeeze<T: PrimeField>(
    state: &mut Vec<T>,
    constants: &Constants<T>,
    output_length: u32,
    r: usize,
) -> Vec<T> {
    let mut output: Vec<T> = Vec::new();

    while (output.len() as u32) < output_length {
        output.extend_from_slice(&state[..r]);
        poseidon_permutation(state, constants);
    }
    if output_length > 1 {
        while output.len() as u32 % output_length != 0 {
            output.pop();
        }
    } else {
        while (output.len() as u32) > 1 {
            output.pop();
        }
    }
    output.clone()
}

/*********************************************************
Absorbing stage
*********************************************************/
fn absorb<T: PrimeField>(input: &Vec<T>, constants: &Constants<T>, r: usize) -> Vec<T> {
    let mut state: Vec<T> = Vec::new();
    let padded_input = pad(input, r as u32);

    init_state(&mut state, constants.t);

    for i in (0..padded_input.len()).step_by(r) {
        add_block(&padded_input[i..i + r], &mut state, r);
        poseidon_permutation(&mut state, constants);
    }
    state.clone()
}

/*********************************************************
Add the inner state with the input slice
*********************************************************/
fn add_block<T: PrimeField>(input: &[T], state: &mut Vec<T>, r: usize) {
    for i in 0..r {
        state[i].add_assign(input[i]);
    }
}

/*********************************************************
Padding function for an input vector.
The functions pads input with 0s and returns a vector
that is a multiple of r. If the length of the input is a
multiple of r, then no padding takes place.
*********************************************************/
fn pad<T: PrimeField>(input: &Vec<T>, r: u32) -> Vec<T> {
    let mut padded_input: Vec<T> = input.to_vec();

    while padded_input.len() as u32 % r != 0 {
        padded_input.push(T::ZERO);
    }

    padded_input
}

/*********************************************************
Implements the poseidon permutation.
*********************************************************/
pub fn poseidon_permutation<T: PrimeField>(state: &mut Vec<T>, constants: &Constants<T>) {
    for i in 0..(constants.full_rounds + constants.partial_rounds) as usize {
        ark(state, constants, i);
        sbox(state, constants, i);
        linear_layer(state, constants);
    }
}

/*********************************************************
Executes de linear layer.
Multiplies the MDS matrix times the state
*********************************************************/
fn linear_layer<T: PrimeField>(state: &mut Vec<T>, constants: &Constants<T>) {
    let mut result: Vec<T> = Vec::new();
    init_state(&mut result, constants.t);

    for i in 0..constants.t {
        for j in 0..constants.t {
            result[i].add_assign(state[j] * constants.m[i][j]);
        }
    }
    *state = result.clone();
}

/*********************************************************
Executes the S-box stage
Computes for each element in the state x^alpha
The rounds are counted starting from 0.
*********************************************************/
fn sbox<T: PrimeField>(state: &mut Vec<T>, constants: &Constants<T>, round_number: usize) {
    if round_number as u32 >= constants.full_rounds / 2
        && (round_number as u32) < constants.full_rounds / 2 + constants.partial_rounds
    {
        // apply partial s-box
        let p: arkBigInt<1> = arkBigInt::from(constants.alpha);
        state[0] = state[0].pow(p);
    } else {
        // apply full s-box
        for i in 0..state.len() {
            let p: arkBigInt<1> = arkBigInt::from(constants.alpha);
            state[i] = state[i].pow(p);
        }
    }
}

/*********************************************************
Executes the ARK stage.
The rounds are counted starting from 0.
*********************************************************/
fn ark<T: PrimeField>(state: &mut Vec<T>, constants: &Constants<T>, round_number: usize) {
    for i in 0..constants.t {
        state[i].add_assign(constants.c[constants.t * round_number + i]);
    }
}

/*********************************************************
Initialize a state vector
**********************************************************/
fn init_state<T: PrimeField>(state: &mut Vec<T>, t: usize) {
    state.clear();
    for _i in 0..t {
        state.push(T::ZERO);
    }
}

/********************************************************
Tests
 *********************************************************/
#[cfg(test)]
mod poseidon_permutation {
    use crate::*;
    use ark_std::UniformRand;

    #[test]
    fn read_constants_files() {
        let constant = read_constants_bls12381_Fq_n255_t5_alpha5_M128_RF8_RP56();
        assert_eq!(
            (constant.partial_rounds + constant.full_rounds) * constant.t as u32,
            constant.c.len() as u32
        );
        assert_eq!(5, constant.m.len());
        assert_eq!(5, constant.m[0].len());
    }

    #[test]
    fn padd_test() {
        let state: Vec<Frbls12_381> = vec![
            Frbls12_381::from(1),
            Frbls12_381::from(2),
            Frbls12_381::from(3),
            Frbls12_381::from(4),
            Frbls12_381::from(5),
            Frbls12_381::from(6),
            Frbls12_381::from(7),
            Frbls12_381::from(8),
        ];

        let new_state = pad(&state, 3);

        assert_eq!(new_state.len(), 9);
    }

    #[test]
    fn ark_test() {
        let mut constants = read_constants_bls12381_Fq_n255_t5_alpha5_M128_RF8_RP56();
        let mut state: Vec<Fqbls12_381> = Vec::new();
        let mut result: Vec<Fqbls12_381> = Vec::new();
        let mut rng = ark_std::test_rng();

        constants.c.clear();

        for i in 0..constants.t {
            state.push(Fqbls12_381::rand(&mut rng));
            constants.c.push(Fqbls12_381::rand(&mut rng));
            result.push(state[i] + constants.c[i]);
        }

        ark(&mut state, &constants, 0);
        assert_eq!(state, result);
    }
}


use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;

use ark_bls12_381::Fq as Fqbls12_381;
use ark_bls12_381::Fr as Frbls12_381;
use ark_bls12_381::Fq as Fqbls12_377;
use ark_bls12_377::Fr as Frbls12_377;
use ark_ff::PrimeField;
use ark_std::str::FromStr;
use core::str;

use num_bigint::BigInt;
use num_traits::Num;

#[derive(Clone, Debug)]
pub struct Constants<T: PrimeField> {
    pub c: Vec<T>,           //round constants
    pub m: Vec<Vec<T>>,      //MDS matrix
    pub t: usize,            //width of the state
    pub partial_rounds: u32, //number of partial rounds
    pub full_rounds: u32,    //number of full rounds
    pub alpha: u32,          //exponent of the S-box
}

/********************************************************
Reads the round constants and MDS matriz from the
given file_name generated by the sage subrutine.
This function read files in the output format of
https://extgit.iaik.tugraz.at/krypto/hadeshash/-/blob/master/code/generate_params_poseidon.sage
 *********************************************************/
 #[allow(non_snake_case)]
 pub fn read_constants_bls12381_Fr_n255_t5_alpha5_M128_RF8_RP56() -> Constants<Frbls12_381> {
     /*
     Params: n=255, t=5, alpha=5, M=128, R_F=8, R_P=56
     Modulus = 52435875175126190479447740508185965837690552500527637822603658699938581184513
     Number of round constants: 320
      */
     let file = File::open("poseidon_params_Frbls12381_n255_t5_alpha5_M128.txt").expect("file not found");
     let reader = BufReader::new(file);
 
     let mut c: Vec<Frbls12_381> = Vec::new();
     let mut m: Vec<Vec<Frbls12_381>> = Vec::new();
 
     let mut i = 0;
 
     for line in reader.lines() {
         // line 5 contains the round constants
         if i == 5 {
             let mut rconst: String = line.unwrap().replace(" ", "").replace("'", "");
             rconst.pop();
             rconst.remove(0);
 
             let constants: Vec<&str> = rconst.split(',').collect();
             for constant in constants {
                 //all constants in the file are writen in hex and need to be converted to dec
                 let n = BigInt::from_str_radix(&constant[2..], 16).unwrap();
                 let number: Frbls12_381 = Frbls12_381::from_str(&n.to_string()).unwrap();
                 c.push(number);
             }
             i += 1;
         }
         // line 18 contains the mds matrix
         else if i == 18 {
             let mut mds = line.unwrap().replace(" ", "").replace("'", "");
             mds.pop();
             mds.pop();
             mds.remove(0);
             mds.remove(0);
             let rows: Vec<&str> = mds.split("],[").collect();
 
             for r in rows {
                 let rows_vector: Vec<&str> = r.split(",").collect();
                 let mut mi: Vec<Frbls12_381> = Vec::new();
                 for r2 in rows_vector {
                     //all constants in the file are writen in hex and need to be converted to dec
                     let n2 = BigInt::from_str_radix(&r2[2..], 16).unwrap();
                     let v2: Frbls12_381 = Frbls12_381::from_str(&n2.to_string()).unwrap();
                     mi.push(v2);
                 }
                 m.push(mi);
             }
             i += 1;
         }
         i += 1;
     }
 
     Constants {
         c,
         m,
         t: 5,
         partial_rounds: 56,
         full_rounds: 8,
         alpha: 5,
     }
 }

/********************************************************
Reads the round constants and MDS matriz from the
given file_name generated by the sage subrutine.
This function read files in the output format of
https://extgit.iaik.tugraz.at/krypto/hadeshash/-/blob/master/code/generate_params_poseidon.sage
 *********************************************************/
#[allow(non_snake_case)]
pub fn read_constants_bls12381_Fq_n255_t5_alpha5_M128_RF8_RP56() -> Constants<Fqbls12_381> {
    /*
    Params: n=255, t=5, alpha=5, M=128, R_F=8, R_P=56
    Modulus = 4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787
    Number of round constants: 320
     */
    let file = File::open("poseidon_params_Fqbls12381_n255_t5_alpha5_M128.txt").expect("file not found");
    let reader = BufReader::new(file);

    let mut c: Vec<Fqbls12_381> = Vec::new();
    let mut m: Vec<Vec<Fqbls12_381>> = Vec::new();

    let mut i = 0;

    for line in reader.lines() {
        // line 5 contains the round constants
        if i == 5 {
            let mut rconst: String = line.unwrap().replace(" ", "").replace("'", "");
            rconst.pop();
            rconst.remove(0);

            let constants: Vec<&str> = rconst.split(',').collect();
            for constant in constants {
                //all constants in the file are writen in hex and need to be converted to dec
                let n = BigInt::from_str_radix(&constant[2..], 16).unwrap();
                let number: Fqbls12_381 = Fqbls12_381::from_str(&n.to_string()).unwrap();
                c.push(number);
            }
            i += 1;
        }
        // line 18 contains the mds matrix
        else if i == 18 {
            let mut mds = line.unwrap().replace(" ", "").replace("'", "");
            mds.pop();
            mds.pop();
            mds.remove(0);
            mds.remove(0);
            let rows: Vec<&str> = mds.split("],[").collect();

            for r in rows {
                let rows_vector: Vec<&str> = r.split(",").collect();
                let mut mi: Vec<Fqbls12_381> = Vec::new();
                for r2 in rows_vector {
                    //all constants in the file are writen in hex and need to be converted to dec
                    let n2 = BigInt::from_str_radix(&r2[2..], 16).unwrap();
                    let v2: Fqbls12_381 = Fqbls12_381::from_str(&n2.to_string()).unwrap();
                    mi.push(v2);
                }
                m.push(mi);
            }
            i += 1;
        }
        i += 1;
    }

    Constants {
        c,
        m,
        t: 5,
        partial_rounds: 56,
        full_rounds: 8,
        alpha: 5,
    }
}


impl MerkleTree {
    /// Constructs a new Merkle tree from the given data.
    ///
    /// This function creates a new Merkle tree from the provided data,
    /// where each element in the data is hashed and stored in the tree.
    ///
    /// # Arguments
    ///
    /// * `data` - A vector containing tuples of addresses and amounts to be stored in the Merkle tree.
    ///
    /// # Returns
    ///
    /// A new instance of `MerkleTree` containing the constructed Merkle tree.
    ///
    /// # Example
    ///
    /// ```rust
    /// use oz_merkle_rs::MerkleTree;
    /// use ethers::types::{Address, U256};
    /// use std::str::FromStr;
    ///
    /// // Create some sample data
    /// let data = vec![
    ///     (Address::from_str("0x1111111111111111111111111111111111111111").unwrap(),
    ///         U256::from_dec_str("1840233889215604334017").unwrap()),
    ///     (Address::from_str("0x00393d62f17b07e64f7cdcdf9bdc2fd925b20bba").unwrap(),
    ///         U256::from_dec_str("7840233889215604334017").unwrap()),
    /// ];
    ///
    /// // Create a new Merkle tree from the data
    /// let merkle_tree = MerkleTree::new(data);
    ///
    pub fn new(data: Vec<(Address, U256)>) -> Self {
        let mut elements: Vec<H256> = data.iter().map(|x| Self::hash_node(*x)).collect();
        // sort and deduplicate to get the correct order of elements
        elements.sort();
        elements.dedup();
        let leaves = elements.len();
        let mut layers = vec![elements.clone()];
        while layers.last().unwrap().len() > 1 {
            layers.push(Self::next_layer(layers.last().unwrap()));
        }
        MerkleTree {
            elements,
            layers,
            leaves,
        }
    }
    /// Retrieves the root hash of the Merkle tree.
    ///
    /// This function returns the root hash of the Merkle tree, if it exists.
    ///
    /// # Returns
    ///
    /// An `Option` containing either the root hash if the Merkle tree is not empty,
    /// or `None` if the Merkle tree is empty.
    pub fn get_root(&self) -> Option<H256> {
        self.layers
            .last()
            .and_then(|last_layer| last_layer.first().cloned())
    }
    /// Retrieves the Merkle proof for a given element.
    ///
    /// This function takes an element and returns the Merkle proof for that element,
    /// if it exists in the Merkle tree.
    ///
    /// # Arguments
    ///
    /// * `element` - The hash of the element for which the proof is to be retrieved.
    ///
    /// # Returns
    ///
    /// An `Option` containing either the Merkle proof as a vector of hashes if the element is found,
    /// or `None` if the element is not present in the Merkle tree.
    pub fn get_proof(&self, element: H256) -> Option<Vec<H256>> {
        let mut index = self.elements.iter().position(|&e| e == element)?;
        let mut proof = Vec::new();

        for layer in &self.layers[..self.layers.len() - 1] {
            let pair_index = if index % 2 == 0 { index + 1 } else { index - 1 };
            if pair_index < layer.len() {
                proof.push(layer[pair_index]);
            }
            index /= 2; // move up to the next layer.
        }
        Some(proof)
    }
    /// Verifies a proof for a given element in a Merkle tree.
    ///
    /// This function takes an element, a proof (list of hashes), and the root hash of the Merkle tree,
    /// and verifies if the element is part of the Merkle tree with the given proof.
    ///
    /// # Arguments
    ///
    /// * `element` - The hash of the element to be verified.
    /// * `proof` - A vector containing the hashes forming the Merkle proof.
    /// * `root` - The root hash of the Merkle tree.
    ///
    /// # Returns
    ///
    /// `true` if the proof is valid for the given element and root hash,
    pub fn verify_proof(&self, element: H256, proof: Vec<H256>, root: H256) -> bool {
        let mut computed_hash = element;

        for proof_element in proof.into_iter() {
            computed_hash = if computed_hash < proof_element {
                Self::hash_pair(&computed_hash, &proof_element)
            } else {
                Self::hash_pair(&proof_element, &computed_hash)
            };
        }
        computed_hash == root
    }
    /// Returns the number of leaves in the Merkle tree.
    ///
    /// This function returns the total number of leaves (i.e., elements) in the Merkle tree.
    ///
    /// # Returns
    ///
    /// The number of leaves (elements) in the Merkle tree.
    ///
    pub fn leaves_length(&self) -> usize {
        self.leaves
    }
    /// Computes the hash of a leaf node in a Merkle tree.
    ///
    /// This function takes the index and leaf data (address and amount) as input,
    /// concatenates them together, and computes the hash of the resulting byte array.
    /// The hash is returned as a `H256` value.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the leaf node in the Merkle tree.
    /// * `leaf_data` - A tuple containing the address (`Address`) and amount (`U256`) of the leaf node.
    ///
    /// # Returns
    ///
    /// A `H256` value representing the hash of the leaf node.
    pub fn hash_node(leaf_data: (Address, U256)) -> H256 {
        let constant = 
            poseidon_lib::read_constants::read_constants_bls12381_Fr_n255_t5_alpha5_M128_RF8_RP56();
        let (account, amount) = leaf_data;
        let mut account_bytes = [0u8; 20];
        account_bytes.copy_from_slice(account.as_bytes());
        let mut amount_bytes = [0u8; 32];
        amount.to_big_endian(&mut amount_bytes);

        let encoded_data =
            ethabi::encode(&[ethabi::Token::Address(account), ethabi::Token::Uint(amount)]);
        let mut encoded_data_381: Vec<Frbls12_381> = Vec::new();
        for i in 0..encoded_data.len(){
            encoded_data_381.push(Frbls12_381::from(encoded_data[i]));
        }
        let hashed_data = poseidon_lib::hash(&encoded_data_381, &constant, 4, 4);
        let mut as_u64 = Vec::new();
        for byte in hashed_data.into_iter().as_ref().iter(){
            as_u64.push(byte.0.as_ref().to_vec());
        }
        let mut bytes = vec![];
        for slice in as_u64 {
            bytes.extend_from_slice(&slice.iter().flat_map(|&x| x.to_le_bytes().to_vec()).collect::<Vec<u8>>());
        }
        
        // Pad to 256 bits
        while bytes.len() < 32 {
            bytes.push(0);
        }
    
        let mut h256_bytes = [0u8; 32];
        h256_bytes.copy_from_slice(&bytes[..32]);
        H256::from(h256_bytes)
    }

    fn next_layer(elements: &[H256]) -> Vec<H256> {
        elements
            .chunks(2)
            .map(|chunk| {
                if chunk.len() == 2 {
                    Self::hash_pair(&chunk[0], &chunk[1])
                } else {
                    // if there are odd layers we hash the last element with itself
                    *chunk.first().unwrap()
                }
            })
            .collect()
    }

    fn hash_pair(a: &H256, b: &H256) -> H256 {
        let mut pairs = [a, b];
        // Ensure lexicographical order
        pairs.sort();
        let concatenated = [pairs[0].as_bytes(), pairs[1].as_bytes()].concat();
        H256::from(keccak256(&concatenated))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::str::FromStr;
    fn setup_tree() -> MerkleTree {
        let data = vec![
            (
                Address::from_str("0x00393d62f17b07e64f7cdcdf9bdc2fd925b20bba").unwrap(),
                U256::from_dec_str("1840233889215604334017").unwrap(),
            ),
            (
                Address::from_str("0x008EF27b8d0B9f8c1FAdcb624ef5FebE4f11fa9f").unwrap(),
                U256::from_dec_str("73750290420694562195").unwrap(),
            ),
        ];
        MerkleTree::new(data)
    }
    #[test]
    fn merkle_tree_creation() {
        let tree = setup_tree();
        assert!(
            !tree.get_root().expect("no root found").is_zero(),
            "The root hash should not be zero"
        );
    }
    #[test]
    fn merkle_tree_root_hash_correctness() {
        let tree = setup_tree();
        let expected_root_hash =
            "0xf699ff5e6437c56f56f6bb1b95c2cf7701b50c9ac75398e7f07ea151e4fee846";

        assert_eq!(
            format!("{:?}", tree.get_root().unwrap()),
            expected_root_hash,
            "The calculated root hash should match the expected value"
        );
    }
    #[test]
    fn get_proof_for_valid_index() {
        let data = (
            Address::from_str("0x00393d62f17b07e64f7cdcdf9bdc2fd925b20bba").unwrap(),
            U256::from_dec_str("1840233889215604334017").unwrap(),
        );
        let tree = setup_tree();
        let proof = tree.get_proof(MerkleTree::hash_node(data)).unwrap();

        assert!(
            !proof.is_empty(),
            "Expected non-empty proof for a valid leaf"
        );
    }
    #[test]
    fn get_proof_for_invalid_index() {
        let data = (
            Address::from_str("0x1111111111111111111111111111111111111111").unwrap(),
            U256::from_dec_str("1840233889215604334017").unwrap(),
        );
        let tree = setup_tree();
        let proof_result = tree.get_proof(MerkleTree::hash_node(data));

        assert!(
            proof_result.is_none(),
            "Expected error when requesting proof for an invalid index"
        );
    }
    #[test]
    fn verify_valid_proof() {
        let data = (
            Address::from_str("0x00393d62f17b07e64f7cdcdf9bdc2fd925b20bba").unwrap(),
            U256::from_dec_str("1840233889215604334017").unwrap(),
        );
        let tree = setup_tree();
        let node = MerkleTree::hash_node(data);
        let proof = tree.get_proof(node).unwrap();
        let result = tree.verify_proof(node, proof, tree.get_root().unwrap());

        assert!(
            result,
            "Proof should be valid and verification should succeed"
        );
    }

    #[test]
    fn verify_valid_proof2() {
        let data = (
            Address::from_str("0x00393d62f17b07e64f7cdcdf9bdc2fd925b20bba").unwrap(),
            U256::from_dec_str("1840233889215604334017").unwrap(),
        );
        let tree = setup_tree();
        let node = MerkleTree::hash_node(data);
        let proof = tree.get_proof(node).unwrap();
        let result = tree.verify_proof(node, proof, tree.get_root().unwrap());

        assert!(
            result,
            "Proof should be valid and verification should succeed"
        );
    }
}
