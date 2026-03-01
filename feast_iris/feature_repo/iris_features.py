from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, String

# Define the Data Source
iris_source = FileSource(
    path="data/iris_dataset.parquet", 
    timestamp_field="event_timestamp"
)

# Define the Entity
iris_entity = Entity(
    name="sample_id", 
    value_type=ValueType.INT64, 
    description="Unique identifier for an Iris flower sample"
)

# Define the Feature View
iris_feature_view = FeatureView(
    name="iris_features",
    entities=[iris_entity],
    ttl=timedelta(weeks=52),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
        Field(name="species", dtype=String), 
    ],
    online=True,
    source=iris_source,
)