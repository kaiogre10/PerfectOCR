# generar_hash.py
import hashlib
import getpass
import secrets
import re
import bycrypt

def validate_password_strength(password):
    """
    Valida la fortaleza de la contraseña.
    """
    if len(password) < 12:
        return False, "La contraseña debe tener al menos 12 caracteres."
    
    if not re.search(r'[A-Z]', password):
        return False, "La contraseña debe contener al menos una mayúscula."
    
    if not re.search(r'[a-z]', password):
        return False, "La contraseña debe contener al menos una minúscula."
    
    if not re.search(r'\d', password):
        return False, "La contraseña debe contener al menos un número."
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "La contraseña debe contener al menos un carácter especial."
    
    return True, "Contraseña válida."

def generate_secure_hash(password, method='bcrypt'):
    """
    Genera un hash seguro usando diferentes métodos.
    """
    if method == 'bcrypt':
        # Método más seguro - automáticamente incluye salt
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    elif method == 'sha256_salt':
        # SHA-256 con salt personalizado
        salt = secrets.token_hex(32)
        password_salt = password + salt
        hashed = hashlib.sha256(password_salt.encode('utf-8')).hexdigest()
        return f"{hashed}:{salt}"
    
    else:
        # Método básico (no recomendado para producción)
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

def main():
    """
    Solicita una contraseña de forma segura y muestra su hash.
    """
    try:
        print("=== Generador de Hash Seguro ===\n")
        
        # Solicitar contraseña
        password = getpass.getpass("Introduce tu contraseña secreta para la API: ")
        if not password:
            print("La contraseña no puede estar vacía.")
            return

        # Validar fortaleza de la contraseña
        is_valid, message = validate_password_strength(password)
        if not is_valid:
            print(f"{message}")
            print("Usa una contraseña más fuerte para mayor seguridad.")
            return

        print("Contraseña válida.")
        
        # Generar hashes con diferentes métodos
        print("\n--- Hashes Generados ---")
        
        # Método recomendado: bcrypt
        try:
            bcrypt_hash = generate_secure_hash(password, 'bcrypt')
            print(f"BCRYPT (RECOMENDADO): {bcrypt_hash}")
        except ImportError:
            print("bcrypt no disponible. Instala con: pip install bcrypt")
        
        # SHA-256 con salt
        sha256_salt_hash = generate_secure_hash(password, 'sha256_salt')
        print(f"SHA-256 + SALT: {sha256_salt_hash}")
        
        # SHA-256 básico (solo para compatibilidad)
        basic_hash = generate_secure_hash(password, 'basic')
        print(f"SHA-256 BÁSICO: {basic_hash}")
        
        print("\n--- Consejos de Seguridad HTTPS ---")
        print("1. Usa certificados SSL/TLS válidos (Let's Encrypt es gratuito)")
        print("2. Implementa autenticación de dos factores (2FA)")
        print("3. Usa tokens JWT con expiración corta (15-30 min)")
        print("4. Implementa rate limiting para prevenir ataques de fuerza bruta")
        print("5. Rota las contraseñas regularmente")
        print("6. Usa HTTPS estricto (HSTS headers)")
        print("7. Implementa logging de seguridad")
        
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    main()