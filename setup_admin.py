from core.database import init_db, create_user
from core.security import hash_password

init_db()
admin_pwd = hash_password("admin123")
create_user("ouidad_admin", admin_pwd, "Admin")
print("Base de données initialisée. Compte 'ouidad_admin' créé avec le mot de passe 'admin123'.")