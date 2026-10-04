use reference_data::{Dtype, Fixture, FixtureError, TensorOrder};

const SAMPLE: &str = include_str!("../../../fixtures/sample.json");

#[test]
fn loads_sample_fixture() {
    let fixture = Fixture::from_json(SAMPLE).unwrap();
    assert_eq!(fixture.name, "matmul_2x2");
    assert_eq!(fixture.metadata.source.as_deref(), Some("hand-computed"));

    let a = fixture.tensor("a").unwrap();
    assert_eq!(a.shape, [2, 2]);
    assert_eq!(a.order, TensorOrder::ColMajor);
    assert_eq!(a.storage.dtype(), Dtype::F64);
    assert_eq!(a.as_f64().unwrap(), &[1.0, 3.0, 2.0, 4.0]);
    assert_eq!(a.element_count(), 4);

    let product = fixture.tensor("product").unwrap();
    assert_eq!(product.as_f64().unwrap(), &[19.0, 43.0, 22.0, 50.0]);

    let mask = fixture.tensor("mask").unwrap();
    assert_eq!(mask.as_bool().unwrap(), &[true, false, true]);
    assert!(mask.as_f64().is_none());

    assert!(fixture.tensor("missing").is_none());
}

#[test]
fn rejects_length_mismatch() {
    let json = r#"{
        "name": "bad",
        "tensors": { "x": { "dtype": "f32", "shape": [2, 2], "data": [1.0] } }
    }"#;
    let err = Fixture::from_json(json).unwrap_err();
    assert!(matches!(err, FixtureError::Format(_)));
}

#[test]
fn rejects_non_finite_as_number_error() {
    // serde_json cannot represent NaN, so this exercises the "not numeric" path
    // with a string in a numeric slot.
    let json = r#"{
        "name": "bad",
        "tensors": { "x": { "dtype": "f64", "shape": [1], "data": ["nope"] } }
    }"#;
    assert!(Fixture::from_json(json).is_err());
}

#[test]
fn rejects_unknown_dtype() {
    let json = r#"{
        "name": "bad",
        "tensors": { "x": { "dtype": "f16", "shape": [1], "data": [1.0] } }
    }"#;
    assert!(matches!(
        Fixture::from_json(json).unwrap_err(),
        FixtureError::Format(_)
    ));
}
