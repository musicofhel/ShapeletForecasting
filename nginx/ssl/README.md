# SSL Certificate Directory

This directory is for storing SSL certificates for HTTPS configuration.

## Setup Instructions

### Option 1: Let's Encrypt (Recommended for Production)

1. Install Certbot:
   ```bash
   sudo apt-get update
   sudo apt-get install certbot
   ```

2. Generate certificates:
   ```bash
   sudo certbot certonly --standalone -d your-domain.com
   ```

3. Copy certificates to this directory:
   ```bash
   sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ./cert.pem
   sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem ./key.pem
   ```

### Option 2: Self-Signed Certificate (Development Only)

Generate a self-signed certificate:
```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout key.pem \
  -out cert.pem \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
```

### Option 3: Commercial Certificate

1. Generate a Certificate Signing Request (CSR):
   ```bash
   openssl req -new -newkey rsa:2048 -nodes \
     -keyout key.pem \
     -out csr.pem
   ```

2. Submit the CSR to your certificate authority

3. Place the received certificate as `cert.pem` in this directory

## File Structure

After setup, this directory should contain:
- `cert.pem` - The SSL certificate (public key)
- `key.pem` - The private key
- `dhparam.pem` (optional) - Diffie-Hellman parameters for enhanced security

## Security Notes

- **Never commit private keys to version control**
- Set appropriate file permissions:
  ```bash
  chmod 600 key.pem
  chmod 644 cert.pem
  ```
- Regularly renew certificates (Let's Encrypt certificates expire after 90 days)

## Nginx Configuration

The nginx.conf file is pre-configured to use these certificates. Uncomment the SSL sections and update the server_name directive with your domain.
