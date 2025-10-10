#!/usr/bin/env bash
set -euo pipefail
mkdir -p infra/certs
openssl req -x509 -nodes -days 365 -newkey rsa:2048   -keyout infra/certs/privkey.pem   -out infra/certs/fullchain.pem   -subj "/C=PL/ST=Mazowieckie/L=Warsaw/O=DataGenius/OU=Dev/CN=${DOMAIN:-localhost}"
echo "Generated self-signed certs in infra/certs"
