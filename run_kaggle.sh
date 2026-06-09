#!/bin/bash

# Dừng script nếu có lỗi xảy ra
set -e

echo "=== 1. Cloning repository ==="
if [ ! -d "RL_Control_traffic_light" ]; then
    git clone https://github.com/vietanhlee/RL_Control_traffic_light
else
    echo "Directory RL_Control_traffic_light already exists, skipping clone."
fi

# Chuyển vào thư mục dự án
cd RL_Control_traffic_light || exit

echo "=== 2. Installing dependencies ==="
pip install -r requirements.txt

echo "=== 3. Installing and configuring PostgreSQL ==="
apt-get update -qq
apt-get install -y postgresql postgresql-contrib

# Khởi động PostgreSQL
service postgresql start

# Đổi mật khẩu và tạo database
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'odoo';"
sudo -u postgres psql -c "DROP DATABASE IF EXISTS traffic_simulator;"
sudo -u postgres psql -c "CREATE DATABASE traffic_simulator;"

# Chỉnh port PostgreSQL thành 5433
sudo sed -i "s/^#port = 5432/port = 5433/" /etc/postgresql/*/main/postgresql.conf
sudo sed -i "s/^port = 5432/port = 5433/" /etc/postgresql/*/main/postgresql.conf

# Khởi động lại PostgreSQL để nhận port mới
service postgresql restart

# Kiểm tra port 5433
pg_isready -h localhost -p 5433

echo "=== 4. Creating .env file ==="
cat <<EOF > .env
# PostgreSQL connection settings for the traffic simulator
# Fill DB_PASSWORD before running the backend.

DB_HOST=localhost
DB_PORT=5433
DB_NAME=traffic_simulator
DB_USER=postgres
DB_PASSWORD=odoo
DB_SSLMODE=prefer

# Optional full connection string. Leave blank to build it from the variables above.
DATABASE_URL=
EOF
echo ".env created successfully."

echo "=== 5. Testing PostgreSQL connection ==="
python -c "
import psycopg2
try:
    conn = psycopg2.connect(
        host='localhost',
        port=5433,
        database='traffic_simulator',
        user='postgres',
        password='odoo'
    )
    print('Connected successfully to PostgreSQL!')
    conn.close()
except Exception as e:
    print('Connection failed:', e)
    exit(1)
"

echo "=== 6. Starting Backend (main.py) ==="
# Khởi chạy main.py dưới dạng process ngầm (background) và đẩy log ra file
python -u main.py > /dev/null 2>&1 &
MAIN_PID=$!
echo "main.py is running in background with PID: $MAIN_PID"

echo "Waiting 120 seconds for the backend to initialize completely..."
sleep 120

echo "=== 7. Starting RL Training ==="
# Chạy script train ở màn hình chính (foreground) để xem output
python -u "RL model/train.py" --steps 36000

echo "=== Training Finished ==="
# Dọn dẹp process backend sau khi train xong
kill $MAIN_PID || true
echo "Backend process terminated."
