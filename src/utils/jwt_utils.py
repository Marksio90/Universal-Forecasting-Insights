from jose import jwt, JWTError
def decode_role(token: str, secret: str)->str|None:
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return payload.get("role")
    except JWTError:
        return None
