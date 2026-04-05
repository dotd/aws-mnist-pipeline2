# How to Run Docker Locally?

## Building the Docker
For building the docker I run:
```
docker buildx build --platform linux/amd64 -t my-cv-model:latest .
```

The `docker buildx build` command above uses the following parameters:

- `buildx`: Docker's BuildKit tool that enables building images for multiple platforms (not just your native architecture).
- `--platform linux/amd64`: This flag tells Docker to build the image for the Linux x86_64 architecture (the same as AWS EC2 GPU instances), even if you're building from a different processor type (such as Apple Silicon/ARM on Mac).
- `-t my-cv-model:latest`: Tags the built image with the name `my-cv-model` and the version `latest` for easy reference later.
- `.`: The dot specifies the build context directory (current directory), which includes your Dockerfile and all necessary files.

**Why use these flags?**  
Building with `buildx` and specifying `--platform linux/amd64` ensures that the Docker image will be compatible with AWS EC2 GPU instances, even if you are developing on a non-x86 machine (like a Mac with an ARM CPU).

You can safely run and test your container locally to verify everything works before pushing to AWS.

## Running the Docker Locally

```bash
docker run --rm -v $(pwd)/data:/workspace/data -v $(pwd)/checkpoints:/workspace/checkpoints my-cv-model:latest
```

### Flags explained

- `--rm`: Automatically removes the container when it exits. Keeps your system clean from stopped containers.
- `-v $(pwd)/data:/workspace/data`: Mounts your local `data/` directory into the container at `/workspace/data`. This way downloaded MNIST data persists on your host and won't be re-downloaded each run.
- `-v $(pwd)/checkpoints:/workspace/checkpoints`: Mounts your local `checkpoints/` directory so trained model files are saved to your host machine, not lost when the container exits.
- `my-cv-model:latest`: The image name and tag to run.

### Passing training arguments

You can override the default training arguments by appending them after the image name:

```bash
docker run --rm -v $(pwd)/data:/workspace/data -v $(pwd)/checkpoints:/workspace/checkpoints my-cv-model:latest python src/train.py --epochs 5 --batch-size 128 --lr 0.0005
```

### Running with GPU (if available)

```bash
docker run --rm --gpus all -v $(pwd)/data:/workspace/data -v $(pwd)/checkpoints:/workspace/checkpoints my-cv-model:latest
```

- `--gpus all`: Gives the container access to all available NVIDIA GPUs. Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed on the host.

## General Docker Commands

```bash
docker images                  # List all images stored locally
docker ps                      # List running containers
docker ps -a                   # List all containers (running + stopped)
docker stop <container_id>     # Stop a running container
docker stop $(docker ps -q)    # Stop all running containers (-q returns only container IDs)
docker rm <container_id>       # Remove a stopped container
docker rmi <image_id>          # Remove an image from local storage
docker system df               # Show disk usage by images, containers, and volumes
docker system prune            # Remove all stopped containers, unused networks, and dangling images
docker logs <container_id>     # Show logs/output of a container
docker exec -it <container_id> bash  # Open a shell inside a running container
```