import os
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

## must equal student issuer secret; no fallback, missing env rejects all
_SECRET_KEY = os.getenv("JWT_SECRET")
_ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/student/login")


class TokenUser(BaseModel):
    id: str
    is_admin: bool = False


## claims only, no db; admin revoke lags until access token expiry
def verify_token(token: str = Depends(oauth2_scheme)) -> TokenUser:
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not _SECRET_KEY:
        raise exc
    try:
        payload = jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise exc
    if payload.get("type") == "refresh" or not payload.get("sub"):
        raise exc
    return TokenUser(id=str(payload["sub"]), is_admin=bool(payload.get("is_admin", False)))


def require_admin_token(user: TokenUser = Depends(verify_token)) -> TokenUser:
    if not user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required.")
    return user
