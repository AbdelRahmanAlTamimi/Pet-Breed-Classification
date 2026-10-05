# Pet Breed Classifier

Run the published CPU inference service and classify a local pet photo in three
commands. Replace `/path/to/pet.jpg` with the image you want to classify.

```bash
docker pull abdelrahmanaltamimi/pet-breed:latest
docker compose -f docker/docker-compose.yml up -d
curl --fail-with-body --retry 30 --retry-delay 1 --retry-all-errors \
  -X POST http://localhost:8000/predict \
  -F "file=@/path/to/pet.jpg"
```

The response includes the predicted breed, species, confidence, top three
breeds, and whether the model is confident or uncertain.
