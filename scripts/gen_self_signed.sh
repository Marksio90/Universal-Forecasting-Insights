#!/usr/bin/env bash
# gen-certs.sh — PRO+++ generator certyfikatów dla Nginx (dev/stage)
# - Tworzy lokalne CA (jeśli włączone) i cert serwera z SAN (DNS/IP).
# - Wypluwa: privkey.pem, fullchain.pem (zgodne z nginx.conf).
# - Idempotentne, bezpieczne uprawnienia, ładne logi.
# - Opcjonalne zaufanie CA do systemowego store (TRUST_CA=1).
# -----------------------------------------------------------------------------
# ENV:
#   CERT_DIR=certs                 # katalog na certy (domyślnie: certs)
#   DOMAINS="localhost,127.0.0.1"  # lista DNS/IP (comma/space)
#   CN=localhost                   # Common Name (domyślnie: pierwszy z DOMAINS)
#   DAYS=365                       # ważność certu
#   KEY_BITS=2048                  # 2048|4096
#   USE_CA=1                       # 1: CA + cert podpisany; 0: self-signed
#   C=PL ST=Mazowieckie L=Warsaw O=DataGenius OU=Dev  # pola DN
#   TRUST_CA=0                     # 1: spróbuj dodać CA do trust store (sudo może być wymagane)
# -----------------------------------------------------------------------------
set -Eeuo pipefail

log()  { printf "\033[1;36m[INFO]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[ERR]\033[0m  %s\n" "$*" >&2; }

# === KONFIG ===
CERT_DIR="${CERT_DIR:-certs}"
mkdir -p "$CERT_DIR"
umask 077

DOMAINS_RAW="${DOMAINS:-localhost 127.0.0.1 ::1}"
# Zamień przecinki na spacje:
DOMAINS_RAW="${DOMAINS_RAW//,/ }"
read -r -a DOMAINS_ARR <<< "$DOMAINS_RAW"
if [ "${#DOMAINS_ARR[@]}" -eq 0 ]; then
  DOMAINS_ARR=("localhost")
fi

CN="${CN:-${DOMAINS_ARR[0]}}"
DAYS="${DAYS:-365}"
KEY_BITS="${KEY_BITS:-2048}"
USE_CA="${USE_CA:-1}"

C="${C:-PL}"; ST="${ST:-Mazowieckie}"; L="${L:-Warsaw}"; O="${O:-DataGenius}"; OU="${OU:-Dev}"

SUBJ="/C=$C/ST=$ST/L=$L/O=$O/OU=$OU/CN=$CN"

# Zbuduj SAN: subjectAltName=DNS:x,IP:y,...
SAN_ENTRIES=()
for d in "${DOMAINS_ARR[@]}"; do
  if [[ "$d" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ || "$d" =~ : ]]; then
    SAN_ENTRIES+=("IP:$d")
  else
    SAN_ENTRIES+=("DNS:$d")
  fi
done
SAN_VAL=$(IFS=, ; echo "${SAN_ENTRIES[*]}")

# Ścieżki
CA_KEY="$CERT_DIR/ca.key"
CA_CRT="$CERT_DIR/ca.pem"
CA_SRL="$CERT_DIR/ca.srl"

SRV_KEY="$CERT_DIR/privkey.pem"
SRV_CSR="$CERT_DIR/server.csr"
SRV_CRT="$CERT_DIR/server.crt"
FULLCHAIN="$CERT_DIR/fullchain.pem"
EXTFILE="$CERT_DIR/openssl.ext"

# === FUNKCJE ===
have_addext() {
  # Sprawdź czy OpenSSL wspiera -addext (LibreSSL starsze na macOS często nie)
  if openssl req -help 2>&1 | grep -q -- "-addext"; then return 0; else return 1; fi
}

make_extfile() {
  cat >"$EXTFILE" <<EOF
subjectAltName=$SAN_VAL
basicConstraints=CA:FALSE
keyUsage=Digital Signature, Key Encipherment
extendedKeyUsage=Server Authentication
EOF
}

gen_self_signed() {
  log "Generuję self-signed cert (SAN: $SAN_VAL)"
  if have_addext; then
    openssl req -x509 -nodes -newkey "rsa:$KEY_BITS" -days "$DAYS" \
      -keyout "$SRV_KEY" -out "$FULLCHAIN" \
      -subj "$SUBJ" -addext "subjectAltName=$SAN_VAL"
  else
    make_extfile
    openssl req -nodes -newkey "rsa:$KEY_BITS" \
      -keyout "$SRV_KEY" -out "$SRV_CSR" -subj "$SUBJ"
    openssl x509 -req -days "$DAYS" -in "$SRV_CSR" -signkey "$SRV_KEY" \
      -out "$FULLCHAIN" -extfile "$EXTFILE"
    rm -f "$SRV_CSR" "$EXTFILE"
  fi
  chmod 600 "$SRV_KEY"
}

gen_ca_if_needed() {
  if [ -f "$CA_KEY" ] && [ -f "$CA_CRT" ]; then
    log "Istniejąca lokalna CA znaleziona → $CA_CRT"
    return
  fi
  log "Tworzę lokalną CA (klucz i cert)..."
  openssl genrsa -out "$CA_KEY" "$KEY_BITS"
  openssl req -x509 -new -nodes -key "$CA_KEY" -sha256 -days 3650 \
    -subj "/C=$C/ST=$ST/L=$L/O=$O/OU=Local CA/CN=$CN CA" \
    -out "$CA_CRT"
  chmod 600 "$CA_KEY"
}

gen_server_signed_by_ca() {
  log "Generuję cert serwera podpisany przez CA (SAN: $SAN_VAL)"
  make_extfile
  openssl genrsa -out "$SRV_KEY" "$KEY_BITS"
  openssl req -new -key "$SRV_KEY" -out "$SRV_CSR" -subj "$SUBJ"
  openssl x509 -req -in "$SRV_CSR" -CA "$CA_CRT" -CAkey "$CA_KEY" -CAcreateserial \
    -out "$SRV_CRT" -days "$DAYS" -sha256 -extfile "$EXTFILE"
  cat "$SRV_CRT" "$CA_CRT" > "$FULLCHAIN"
  rm -f "$SRV_CSR" "$EXTFILE" "$CA_SRL"
  chmod 600 "$SRV_KEY"
}

trust_ca() {
  [ "${TRUST_CA:-0}" = "1" ] || { warn "TRUST_CA=1 nie ustawione — pomijam instalację CA w systemie"; return 0; }
  if [ ! -f "$CA_CRT" ]; then warn "Brak CA ($CA_CRT) — nic do zaufania"; return 0; fi
  os="$(uname -s)"
  case "$os" in
    Darwin)
      if command -v security >/dev/null 2>&1; then
        warn "Dodaję CA do System Keychain (wymaga uprawnień admina)"
        sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain "$CA_CRT" || \
          security add-trusted-cert -d -r trustRoot -k "$HOME/Library/Keychains/login.keychain-db" "$CA_CRT" || true
      fi
      ;;
    Linux)
      if [ -d "/usr/local/share/ca-certificates" ]; then
        warn "Instaluję CA w /usr/local/share/ca-certificates (sudo wymagane)"
        sudo cp "$CA_CRT" "/usr/local/share/ca-certificates/datagenius-local-ca.crt"
        sudo update-ca-certificates || true
      elif [ -d "/etc/pki/ca-trust/source/anchors" ]; then
        warn "Instaluję CA w /etc/pki/ca-trust/source/anchors (sudo wymagane)"
        sudo cp "$CA_CRT" "/etc/pki/ca-trust/source/anchors/datagenius-local-ca.crt"
        sudo update-ca-trust || true
      else
        warn "Nieznana dystrybucja – zainstaluj CA ręcznie."
      fi
      ;;
    *)
      warn "OS $os – pomiń lub zainstaluj CA ręcznie."
      ;;
  esac
}

print_summary() {
  log "Gotowe certy w: $CERT_DIR"
  printf "  - privkey : %s\n" "$SRV_KEY"
  printf "  - fullchain: %s\n" "$FULLCHAIN"
  if [ -f "$CA_CRT" ]; then printf "  - CA      : %s\n" "$CA_CRT"; fi
  if command -v openssl >/dev/null 2>&1; then
    echo "Podsumowanie certu:"
    openssl x509 -in "$FULLCHAIN" -noout -subject -issuer -dates -ext subjectAltName || true
    echo "Fingerprint (SHA256):"
    openssl x509 -in "$FULLCHAIN" -noout -fingerprint -sha256 | sed 's/SHA256 Fingerprint=//'
  fi
  echo
  echo "Użycie w Docker/NGINX:"
  echo "  volumes:"
  echo "    - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro"
  echo "    - ./certs:/etc/nginx/certs:ro"
}

# === MAIN ===
if ! command -v openssl >/dev/null 2>&1; then
  err "Brak 'openssl' w PATH"; exit 1
fi

log "Generacja certów dla: ${DOMAINS_ARR[*]}"
log "Tryb: $([ "$USE_CA" = "1" ] && echo 'CA-signed' || echo 'self-signed'), dni: $DAYS, bity: $KEY_BITS"

if [ "$USE_CA" = "1" ]; then
  gen_ca_if_needed
  gen_server_signed_by_ca
else
  gen_self_signed
fi

trust_ca
print_summary
