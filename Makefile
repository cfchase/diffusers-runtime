build:
	podman build -t quay.io/cfchase/diffusers-runtime:latest -f docker/Dockerfile .

push:
	podman push quay.io/cfchase/diffusers-runtime:latest

run:
	podman run -ePORT=8080 -p8080:8080 quay.io/cfchase/diffusers-runtime:latest

test-request:
	curl -H "Content-Type: application/json" localhost:8080/v1/models/model:predict -d @./scripts/input.json
