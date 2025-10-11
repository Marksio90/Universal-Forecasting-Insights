from __future__ import annotations
# === JWT SAFE DECODE (PRO+++) ===
# Rola: bezpieczne dekodowanie roli z JWT z ochroną przed alg-confusion,
# wsparciem dla rotacji kluczy (kid), precyzyjną walidacją claims oraz wynikowym statusem.
# Zgodne wstecznie: decode_role(...) zwraca Optional[str].
from dataclasses import dataclass
from typing import Optional, Tuple, Set, Dict, Any, Callable

from jose import jwt
from jose.exceptions import JWTError, ExpiredSignatureError, JWTClaimsError, JWKError

__all__ = ["JWTDecodeConfig", "JWTDecodeResult", "decode_role_ex", "decode_role"]


# === NAZWA_SEKCJI === KONFIGURACJA ===
@dataclass(frozen=True)
class JWTDecodeConfig:
    """
    Konfiguracja walidacji JWT.
    - algorithms: dozwolone algorytmy (brak 'none'!).
    - issuer/audience: opcjonalna twarda walidacja iss/aud.
    - leeway: tolerancja czasowa (sekundy) dla exp/nbf/iat.
    - verify_*: kontrola weryfikacji standardowych pól.
    - allowed_roles: jeśli podane, rola musi należeć do zbioru.
    - role_claim: nazwa pola z rolą (np. 'role' / 'realm_access.roles' — patrz role_resolver).
    - jwk_keys: mapa {kid: public_key/secret}; preferowana, gdy nagłówek zawiera kid.
    - key_resolver: callback(header, claims) -> key (np. do pobrania JWKS z cache).
    - require_typ: jeżeli ustawione, nagłówek 'typ' musi być jednym z podanych (np. {"JWT","at+jwt"}).
    """
    algorithms: Tuple[str, ...] = ("HS256", "RS256")
    issuer: Optional[str] = None
    audience: Optional[str] = None
    leeway: int = 0
    verify_exp: bool = True
    verify_iat: bool = True
    verify_nbf: bool = True
    allowed_roles: Optional[Set[str]] = None
    role_claim: str = "role"

    jwk_keys: Optional[Dict[str, str]] = None
    key_resolver: Optional[Callable[[Dict[str, Any], Optional[Dict[str, Any]]], str]] = None

    require_typ: Optional[Set[str]] = frozenset({"JWT", "at+jwt"})


# === NAZWA_SEKCJI === WYNIK ===
@dataclass(frozen=True)
class JWTDecodeResult:
    ok: bool
    role: Optional[str]
    claims: Optional[Dict[str, Any]]
    header: Optional[Dict[str, Any]]
    algorithm: Optional[str]
    kid: Optional[str]
    error: Optional[str]


# === NAZWA_SEKCJI === HELPERY ===
def _resolve_key(
    default_key: Optional[str],
    header: Dict[str, Any],
    claims: Optional[Dict[str, Any]],
    cfg: JWTDecodeConfig,
) -> Optional[str]:
    # 1) kid → jwk_keys
    kid = header.get("kid")
    if kid and cfg.jwk_keys and kid in cfg.jwk_keys:
        return cfg.jwk_keys[kid]
    # 2) key_resolver callback (może pobrać z JWKS cache)
    if cfg.key_resolver:
        try:
            key = cfg.key_resolver(header, claims)
            if key:
                return key
        except Exception:
            return None
    # 3) fallback: default_key (np. HS256 shared secret)
    return default_key


def _extract_role(claims: Dict[str, Any], role_claim: str) -> Optional[str]:
    """
    Obsługa prostych i zagnieżdżonych ścieżek (np. 'realm_access.roles[0]').
    Dla list ról zwraca pierwszą (lub None).
    """
    if not role_claim:
        return None

    # Prosta ścieżka bez kropki
    if "." not in role_claim and "[" not in role_claim:
        val = claims.get(role_claim)
        if isinstance(val, str):
            return val
        if isinstance(val, (list, tuple)) and val and isinstance(val[0], str):
            return val[0]
        return None

    # Zagnieżdżona ścieżka
    node: Any = claims
    # Parsowanie proste: split po '.' i obsługa indeksów [i]
    import re
    parts = role_claim.split(".")
    idx_re = re.compile(r"([^\[]+)(\[(\d+)\])?")
    for part in parts:
        m = idx_re.fullmatch(part)
        if not m:
            return None
        key = m.group(1)
        idx = m.group(3)
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
        if idx is not None:
            try:
                i = int(idx)
                if not isinstance(node, (list, tuple)) or i >= len(node):
                    return None
                node = node[i]
            except Exception:
                return None
    if isinstance(node, str):
        return node
    if isinstance(node, (list, tuple)) and node and isinstance(node[0], str):
        return node[0]
    return None


# === NAZWA_SEKCJI === GŁÓWNE API ===
def decode_role_ex(token: str, key: Optional[str] = None, config: Optional[JWTDecodeConfig] = None) -> JWTDecodeResult:
    """
    Bezpiecznie dekoduje JWT i wydobywa rolę.
    - Odrzuca nagłówki z nieobsługiwanym alg i nietypowym 'typ' (jeśli require_typ ustawione).
    - Wspiera rotację kluczy przez 'kid' (cfg.jwk_keys) lub key_resolver().
    - Weryfikuje exp/iat/nbf/iss/aud zgodnie z konfiguracją.
    - Zwraca pełny wynik z informacją o błędzie.

    Uwaga: gdy alg RS*, `key` / `jwk_keys` / `key_resolver` powinny dostarczać KLUCZ PUBLICZNY.
    """
    cfg = config or JWTDecodeConfig()
    header: Optional[Dict[str, Any]] = None
    try:
        header = jwt.get_unverified_header(token)
    except JWTError:
        return JWTDecodeResult(False, None, None, None, None, None, "invalid_header")

    alg = header.get("alg")
    if not isinstance(alg, str) or alg.upper() not in cfg.algorithms or alg.lower() == "none":
        return JWTDecodeResult(False, None, None, header, alg, header.get("kid"), "unsupported_alg")

    typ = header.get("typ")
    if cfg.require_typ and typ and typ not in cfg.require_typ:
        return JWTDecodeResult(False, None, None, header, alg, header.get("kid"), "invalid_typ")

    options = {
        "verify_signature": True,
        "verify_exp": cfg.verify_exp,
        "verify_iat": cfg.verify_iat,
        "verify_nbf": cfg.verify_nbf,
        # verify_aud: jose wymaga aud tylko jeśli podasz audience – pozostawiamy domyślne
        "require_exp": cfg.verify_exp,
    }

    # Klucz (może zależeć od 'kid')
    chosen_key = _resolve_key(key, header, None, cfg)
    if not chosen_key:
        # Przy RS* brak klucza publicznego = błąd
        return JWTDecodeResult(False, None, None, header, alg, header.get("kid"), "no_key")

    try:
        claims = jwt.decode(
            token,
            chosen_key,
            algorithms=list(cfg.algorithms),
            audience=cfg.audience,
            issuer=cfg.issuer,
            options=options,
            leeway=cfg.leeway,
        )
    except ExpiredSignatureError:
        return JWTDecodeResult(False, None, None, header, alg, header.get("kid"), "token_expired")
    except JWTClaimsError as e:
        return JWTDecodeResult(False, None, None, header, alg, header.get("kid"), f"claims_error:{str(e)}")
    except (JWKError, JWTError) as e:
        return JWTDecodeResult(False, None, None, header, alg, header.get("kid"), f"jwt_error:{str(e)}")

    # (opcjonalna) walidacja dodatkowych claims
    role = _extract_role(claims, cfg.role_claim)
    if role is not None and cfg.allowed_roles and role not in cfg.allowed_roles:
        return JWTDecodeResult(False, None, claims, header, alg, header.get("kid"), "role_not_allowed")

    return JWTDecodeResult(True, role, claims, header, alg, header.get("kid"), None)


# === NAZWA_SEKCJI === KOMPATYBILNOŚĆ (stary interfejs) ===
def decode_role(token: str, key: str, config: Optional[JWTDecodeConfig] = None) -> Optional[str]:
    """
    Zgodne wstecznie API: zwraca tylko rolę lub None.
    """
    return decode_role_ex(token, key=key, config=config).role
