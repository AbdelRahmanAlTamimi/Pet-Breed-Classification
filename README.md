Run the published CPU inference service with Docker.

```bash
docker pull abdelrahmanaltamimi/pet-breed:latest
docker tag abdelrahmanaltamimi/pet-breed:latest pet-breed:latest
docker compose -f docker/docker-compose.yml up -d
```
