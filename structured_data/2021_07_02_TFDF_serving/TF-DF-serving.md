# Serving TF-DF models using TF Serving

*This is a small excerpt from our [blogpost on decision forests](https://blog.ml6.eu/serving-decision-forests-with-tensorflow-b447ea4fc81c)*

TensorFlow and TensorFlow Extended are a broad ecosystem of software packages, allowing to train, serve,... ML models.

The [addition of decision forests to TensorFlow](https://www.tensorflow.org/decision_forests) increases the applicability of TensorFlow on many more use cases.

Training is only one part of ML models: robust inference calculations are as important in a data pipeline.
This is where TensorFlow Serving comes in. TensorFlow Serving basically creates an API around a model. On top of that it allows to split code and model, enabling separate versioning.

Unfortunately, when TensorFlow Decision Forests (TF-DF) was released, TensorFlow Serving was not ready to serve these new models.

We decided to build a custom `tf-serving` docker image to serve these models.

The docker image can be found on [DockerHub](https://hub.docker.com/r/ml6team/tf-serving-tfdf).
The model used in the following example was trained using the [TF-DF penguins colab](https://colab.research.google.com/github/tensorflow/decision-forests/blob/main/documentation/tutorials/beginner_colab.ipynb).

To use the docker image for a penguin model, the following code can be used:

```sh
docker run -t --rm \
  -p 8501:8501 \
  -v "/path/to/model:/models/penguin_model" \
  -e MODEL_NAME=insert_model_name \
  ml6team/tf-serving-tfdf:latest
```

For inference calculation the following python code can be used.

```python3
import requests
host = '0.0.0.0'
port = '8501'
data = {
        "year": [[1], [12]],
        "sex": [["male"], ["female"]],
        "island": [["Torgersen"], ["Biscoe"]],
        "body_mass_g": [[4000], [3600]],
        "flipper_length_mm": [[180], [210]],
        "bill_depth_mm": [[17.5], [16.2]],
        "bill_length_mm": [[30], [50]]
        }
json = {"inputs": data}
if __name__ == '__main__':
    server_url = 'http://' + host + ':' + port + '/v1/models/penguin_model:predict'
    response = requests.post(server_url, json=json)
    print(response.text)
```

Enjoy!