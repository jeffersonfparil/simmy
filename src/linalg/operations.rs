use anyhow::{Result, bail};

// See wgsl/opcodes.wgsl for reference and make sure of the consistencies between that and this file!

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    ABS,
    NEG,
    SQRT,
    EXP,
    LOG,
    SIN,
    COS,
    NOT, // logical (not bitwise)
    ADD,
    SUB,
    MUL,
    DIV,
    MIN,
    MAX,
    POW,
    ATAN2,
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
    AND, // logical (not bitwise)
    OR,  // logical (not bitwise)
    XOR, // logical (not bitwise)
}

impl Operation {
    pub fn unary_opcode(self) -> Result<u32> {
        match self {
            Self::ABS => Ok(0),
            Self::NEG => Ok(1),
            Self::SQRT => Ok(2),
            Self::EXP => Ok(3),
            Self::LOG => Ok(4),
            Self::SIN => Ok(5),
            Self::COS => Ok(6),
            Self::NOT => Ok(7),
            _ => bail!("Invalid unary matrix operation!"),
        }
    }
    pub fn binary_opcode(self) -> Result<u32> {
        match self {
            Self::ADD => Ok(0),
            Self::SUB => Ok(1),
            Self::MUL => Ok(2),
            Self::DIV => Ok(3),
            Self::MIN => Ok(4),
            Self::MAX => Ok(5),
            Self::POW => Ok(6),
            Self::ATAN2 => Ok(7),
            Self::EQ => Ok(8),
            Self::NE => Ok(9),
            Self::LT => Ok(10),
            Self::LE => Ok(11),
            Self::GT => Ok(12),
            Self::GE => Ok(13),
            Self::AND => Ok(14),
            Self::OR => Ok(15),
            Self::XOR => Ok(16),
            _ => bail!("Invalid binary matrix operation!"),
        }
    }
    pub fn contract_pairwise_opcode(self) -> Result<u32> {
        match self {
            Self::ADD => Ok(0),
            Self::SUB => Ok(1),
            Self::MUL => Ok(2),
            Self::DIV => Ok(3),
            Self::MIN => Ok(4),
            Self::MAX => Ok(5),
            Self::EQ => Ok(6),
            Self::NE => Ok(7),
            Self::LT => Ok(8),
            Self::LE => Ok(9),
            Self::GT => Ok(10),
            Self::GE => Ok(11),
            Self::AND => Ok(12),
            Self::OR => Ok(13),
            Self::XOR => Ok(14),
            _ => bail!("Invalid contract matrix pairwise operation!"),
        }
    }
    pub fn contract_reduction_opcode(self) -> Result<u32> {
        match self {
            Self::ADD => Ok(0),
            Self::MUL => Ok(1),
            Self::MIN => Ok(2),
            Self::MAX => Ok(3),
            Self::AND => Ok(4),
            Self::OR => Ok(5),
            Self::XOR => Ok(6),
            _ => bail!("Invalid contract matrix reduction operation!"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    ////////////////////////////////////
    // Unary operations
    ////////////////////////////////////
    #[test]
    fn maps_unary_opcodes() {
        assert_eq!(Operation::ABS.unary_opcode().unwrap(), 0);
        assert_eq!(Operation::NEG.unary_opcode().unwrap(), 1);
        assert_eq!(Operation::SQRT.unary_opcode().unwrap(), 2);
        assert_eq!(Operation::EXP.unary_opcode().unwrap(), 3);
        assert_eq!(Operation::LOG.unary_opcode().unwrap(), 4);
        assert_eq!(Operation::SIN.unary_opcode().unwrap(), 5);
        assert_eq!(Operation::COS.unary_opcode().unwrap(), 6);
        assert_eq!(Operation::NOT.unary_opcode().unwrap(), 7);
    }
    #[test]
    fn rejects_invalid_unary_operations() {
        assert!(Operation::ADD.unary_opcode().is_err());
        assert!(Operation::MUL.unary_opcode().is_err());
        assert!(Operation::EQ.unary_opcode().is_err());
        assert!(Operation::AND.unary_opcode().is_err());
    }
    ////////////////////////////////////
    // Binary operations
    ////////////////////////////////////
    #[test]
    fn maps_binary_opcodes() {
        assert_eq!(Operation::ADD.binary_opcode().unwrap(), 0);
        assert_eq!(Operation::SUB.binary_opcode().unwrap(), 1);
        assert_eq!(Operation::MUL.binary_opcode().unwrap(), 2);
        assert_eq!(Operation::DIV.binary_opcode().unwrap(), 3);
        assert_eq!(Operation::MIN.binary_opcode().unwrap(), 4);
        assert_eq!(Operation::MAX.binary_opcode().unwrap(), 5);
        assert_eq!(Operation::POW.binary_opcode().unwrap(), 6);
        assert_eq!(Operation::ATAN2.binary_opcode().unwrap(), 7);
        assert_eq!(Operation::EQ.binary_opcode().unwrap(), 8);
        assert_eq!(Operation::NE.binary_opcode().unwrap(), 9);
        assert_eq!(Operation::LT.binary_opcode().unwrap(), 10);
        assert_eq!(Operation::LE.binary_opcode().unwrap(), 11);
        assert_eq!(Operation::GT.binary_opcode().unwrap(), 12);
        assert_eq!(Operation::GE.binary_opcode().unwrap(), 13);
        assert_eq!(Operation::AND.binary_opcode().unwrap(), 14);
        assert_eq!(Operation::OR.binary_opcode().unwrap(), 15);
        assert_eq!(Operation::XOR.binary_opcode().unwrap(), 16);
    }
    #[test]
    fn rejects_invalid_binary_operations() {
        assert!(Operation::ABS.binary_opcode().is_err());
        assert!(Operation::NEG.binary_opcode().is_err());
        assert!(Operation::SQRT.binary_opcode().is_err());
        assert!(Operation::NOT.binary_opcode().is_err());
    }
    ////////////////////////////////////
    // Contraction pairwise operations
    ////////////////////////////////////
    #[test]
    fn maps_contract_pairwise_opcodes() {
        assert_eq!(Operation::ADD.contract_pairwise_opcode().unwrap(), 0);
        assert_eq!(Operation::SUB.contract_pairwise_opcode().unwrap(), 1);
        assert_eq!(Operation::MUL.contract_pairwise_opcode().unwrap(), 2);
        assert_eq!(Operation::DIV.contract_pairwise_opcode().unwrap(), 3);
        assert_eq!(Operation::MIN.contract_pairwise_opcode().unwrap(), 4);
        assert_eq!(Operation::MAX.contract_pairwise_opcode().unwrap(), 5);
        assert_eq!(Operation::EQ.contract_pairwise_opcode().unwrap(), 6);
        assert_eq!(Operation::NE.contract_pairwise_opcode().unwrap(), 7);
        assert_eq!(Operation::LT.contract_pairwise_opcode().unwrap(), 8);
        assert_eq!(Operation::LE.contract_pairwise_opcode().unwrap(), 9);
        assert_eq!(Operation::GT.contract_pairwise_opcode().unwrap(), 10);
        assert_eq!(Operation::GE.contract_pairwise_opcode().unwrap(), 11);
        assert_eq!(Operation::AND.contract_pairwise_opcode().unwrap(), 12);
        assert_eq!(Operation::OR.contract_pairwise_opcode().unwrap(), 13);
        assert_eq!(Operation::XOR.contract_pairwise_opcode().unwrap(), 14);
    }
    #[test]
    fn rejects_invalid_contract_pairwise_operations() {
        assert!(Operation::ABS.contract_pairwise_opcode().is_err());
        assert!(Operation::NEG.contract_pairwise_opcode().is_err());
        assert!(Operation::SQRT.contract_pairwise_opcode().is_err());
        assert!(Operation::POW.contract_pairwise_opcode().is_err());
        assert!(Operation::ATAN2.contract_pairwise_opcode().is_err());
        assert!(Operation::NOT.contract_pairwise_opcode().is_err());
    }
    ////////////////////////////////////
    // Contraction reduction operations
    ////////////////////////////////////
    #[test]
    fn maps_contract_reduction_opcodes() {
        assert_eq!(Operation::ADD.contract_reduction_opcode().unwrap(), 0);
        assert_eq!(Operation::MUL.contract_reduction_opcode().unwrap(), 1);
        assert_eq!(Operation::MIN.contract_reduction_opcode().unwrap(), 2);
        assert_eq!(Operation::MAX.contract_reduction_opcode().unwrap(), 3);
        assert_eq!(Operation::AND.contract_reduction_opcode().unwrap(), 4);
        assert_eq!(Operation::OR.contract_reduction_opcode().unwrap(), 5);
        assert_eq!(Operation::XOR.contract_reduction_opcode().unwrap(), 6);
    }
    #[test]
    fn rejects_invalid_contract_reduction_operations() {
        assert!(Operation::ABS.contract_reduction_opcode().is_err());
        assert!(Operation::NEG.contract_reduction_opcode().is_err());
        assert!(Operation::DIV.contract_reduction_opcode().is_err());
        assert!(Operation::POW.contract_reduction_opcode().is_err());
        assert!(Operation::EQ.contract_reduction_opcode().is_err());
        assert!(Operation::LT.contract_reduction_opcode().is_err());
        assert!(Operation::NOT.contract_reduction_opcode().is_err());
    }
}
