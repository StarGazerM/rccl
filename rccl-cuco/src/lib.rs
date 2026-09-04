//! Rust-native concurrent GPU containers modeled on NVIDIA cuCollections.
//!
//! [`StaticMap`] provides the packed fixed-capacity map path. [`ExactRowSet`]
//! stores complete HISA [`Row`] keys and verifies full equality after XxHash64
//! filtering. [`BloomFilter`] is reserved for the audited probabilistic
//! prefilter path and is never used for exact deduplication.

mod backend;
mod bloom;
mod bloom_backend;
mod bloom_device;
mod device;
mod dynamic;
mod error;
mod exact_set;
mod exact_set_backend;
mod exact_set_device;
mod hash;
mod kernel_args;
mod macros;
mod map;
mod probing;
mod program;
mod range;
mod word;

#[doc(hidden)]
pub mod __private;

pub use bloom::{
    BLOOM_HASH_COUNT, BLOOM_WORDS_PER_BLOCK, BloomError, BloomFilter, BloomFilterConfig,
    BloomFilterReadRef, BloomFilterRef, HISA_BLOOM_BLOCKS, PendingBloomOperation,
};
pub use dynamic::{DynamicHashMap, DynamicMapConfig};
pub use error::MapError;
pub use exact_set::{
    EXACT_SET_COOPERATIVE_GROUP, ExactInsertOutcome, ExactRowSet, ExactRowSetConfig,
    ExactRowSetReadRef, ExactRowSetRef, ExactSetError, ExactSetInsertReport,
    PendingExactSetOperation, Row,
};
pub use hash::{StableHashBytes, XxHash64};
pub use map::{
    PendingInsert, Sentinels, StaticHashMap, StaticMap, StaticMapConfig, StaticMapReadRef,
    StaticMapRef,
};
pub use probing::{DoubleHashing, LinearProbing, ProbingScheme, XxHash32};
pub use program::{Launch, Program};
pub use range::{Entries, entries};
pub use word::{MapKey, MapValue};

pub use rccl_memory as memory;
