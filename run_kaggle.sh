#!/bin/bash

# Dừng script nếu có lỗi xảy ra
set -e

echo "=== 3. Installing dependencies ==="
pip install -r requirements.txt

echo "=== 4. Installing and configuring PostgreSQL ==="
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

echo "=== 5. Creating .env file ==="
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

echo "=== 6. Testing PostgreSQL connection ==="
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
echo "=== 7. Starting Backend (main.py) ==="
# Chạy backend từ đúng thư mục backend/app như bạn yêu cầu.
cd backend/app || exit
python -u main.py > /dev/null 2>&1 &
MAIN_PID=$!
echo "main.py is running in background with PID: $MAIN_PID"

cd ../..

echo "Waiting 120 seconds for the backend to initialize completely..."
sleep 120

echo "=== 8. Starting RL Training ==="
# Chạy train từ đúng thư mục rl_agent như bạn yêu cầu.
cd rl_agent || exit
python -u train.py --steps 50000

echo "=== Training Finished ==="
# Dọn dẹp process backend sau khi train xong
kill $MAIN_PID || true
echo "Backend process terminated."
