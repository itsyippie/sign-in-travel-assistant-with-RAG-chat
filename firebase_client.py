import math
import time
import requests
from config import FIREBASE_API_KEY, AUTH_TIMEOUT


class FirebaseError(Exception):
    def __init__(self, message, sign_in_again=False):
        super().__init__(message)
        self.sign_in_again = sign_in_again


def validate_config():
    if not FIREBASE_API_KEY:
        raise FirebaseError('Fill FIREBASE_API_KEY in .env, then restart Streamlit.')


def _authenticate(action, email, password):
    validate_config()
    email = email.strip()
    if not email or not password:
        raise FirebaseError('Enter both email and password.')
    try:
        response = requests.post(
            f'https://identitytoolkit.googleapis.com/v1/accounts:{action}',
            params={'key': FIREBASE_API_KEY},
            json={'email': email, 'password': password, 'returnSecureToken': True},
            timeout=AUTH_TIMEOUT,
        )
    except requests.RequestException:
        raise FirebaseError('Cannot reach Firebase. Check your connection and try again.') from None
    try:
        data = response.json()
    except ValueError:
        raise FirebaseError('Firebase returned an unexpected response. Try again later.') from None
    if not isinstance(data, dict):
        raise FirebaseError('Firebase returned an unexpected response.')
    if response.ok:
        return data
    error = data.get('error', {})
    code = str(error.get('message', '')) if isinstance(error, dict) else ''
    messages = {
        'EMAIL_EXISTS': 'That email is already registered. Choose Sign in.',
        'OPERATION_NOT_ALLOWED': 'Enable Email/Password in Firebase Authentication.',
        'WEAK_PASSWORD': 'Choose a password that meets the Firebase password policy.',
        'PASSWORD_DOES_NOT_MEET_REQUIREMENTS': 'Choose a password that meets the Firebase password policy.',
        'INVALID_EMAIL': 'Enter a valid email address.',
        'INVALID_LOGIN_CREDENTIALS': 'Sign-in failed. Check your email and password.',
        'INVALID_PASSWORD': 'Sign-in failed. Check your email and password.',
        'EMAIL_NOT_FOUND': 'Sign-in failed. Check your email and password.',
        'USER_DISABLED': 'This account has been disabled.',
        'TOO_MANY_ATTEMPTS': 'Too many attempts. Wait a little and try again.',
    }
    for marker, message in messages.items():
        if marker in code:
            raise FirebaseError(message)
    if response.status_code == 429:
        raise FirebaseError('Too many requests. Wait a little and try again.')
    # Never expose raw responses, URLs containing keys, or credentials.
    raise FirebaseError('Firebase could not complete the request. Check your API key and Authentication settings.')


def create_account(email, password):
    _authenticate('signUp', email, password)


def sign_in(email, password):
    data = _authenticate('signInWithPassword', email, password)
    try:
        lifetime = int(data['expiresIn'])
        user = {key: data[field] for key, field in (
            ('uid', 'localId'), ('email', 'email'), ('id_token', 'idToken'))}
        if lifetime <= 0 or not all(isinstance(v, str) and v for v in user.values()):
            raise ValueError
        user['expires_at'] = time.time() + lifetime
    except (KeyError, ValueError, TypeError, OverflowError):
        raise FirebaseError('Sign-in returned incomplete data. Please try again.') from None
    return user


def require_session(user):
    """Check local session expiry before chat; no refresh token is retained."""
    try:
        expires = float(user['expires_at'])
        valid = math.isfinite(expires) and time.time() < expires
        valid = valid and all(user.get(k) for k in ('uid', 'email', 'id_token'))
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise FirebaseError('Your session has expired. Please sign in again.', True)
