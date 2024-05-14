build:
	podman build -t quay.io/cfchase/diffusers-runtime:latest -f docker/Dockerfile .

push:
	podman push quay.io/cfchase/diffusers-runtime:latest

run:
	podman run -ePORT=8080 -p8080:8080 quay.io/cfchase/diffusers-runtime:latest
