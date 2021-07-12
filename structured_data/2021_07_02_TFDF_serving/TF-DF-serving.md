# Serving TF-DF models using TF Serving

*This is a small excerpt from the  https://blog.ml6.eu/serving-decision-forests-with-tensorflow-b447ea4fc81c blogpost*

TensorFlow combined with TensorFlow Extended, is a broad ecosystem of software packages, allowing to train, serve,... ML models.

The addition of decision forests to TensorFlow, allows to use TensorFlow on even more use cases.

Training is only one part of ML models. Using it for inference calculations is an equally important part of data pipeline development.
This is where TensorFlow Serving (part of TensorFlow Extended) comes in. TensorFlow serving basically creates an API around a model, and allows to split code and model versioning.

Unfortunately, when TensorFlow Decision Forests (TF-DF) was released, TensorFlow Serving was not ready to except these new models.

As we found TF-DF models to be very interesting, we decided to build a custom docker image to serve these models.

The docker image can be found at (https://hub.docker.com/r/ml6team/tf-serving-tfdf)[https://hub.docker.com/r/ml6team/tf-serving-tfdf].

To use the docker image for a penguin model, the following code can be used:

```sh
docker run -t --rm \
  -p 8501:8501 \
  -v "/path/to/model:/models/model_name" \
  -e MODEL_NAME=insert_model_name \
  ml6team/tf-serving-tfdf:latest
```
To train a TF-DF model, please checkout the (docs)[https://www.tensorflow.org/decision_forests].

For inference calculation the following python code can be used:

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
    server_url = 'http://' + host + ':' + port + '/v1/models/insert_model_name_here:predict'
    response = requests.post(server_url, json=json)
    print(response.text)
```

The data part is very model specific, and needs to modified.