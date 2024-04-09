#!/bin/bash
name=$(whoami)
echo "$name"

export devicename="$name"

docker-compose pull
docker-compose up -d