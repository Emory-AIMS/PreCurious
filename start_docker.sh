docker build ./ -t precurious:latest
docker run --rm \
	--user root \
	--mount type=bind,source=$(readlink -f ./),target=/workspace \
	--gpus all --network host \
	--name precurious_dev -it precurious:latest bash