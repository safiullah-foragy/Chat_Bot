# Deploy to Render.com

## 🚀 Step-by-Step Deployment Guide

### 1. Prepare Your Repository

✅ **Already Done:**
- `render.yaml` - Render configuration
- `Procfile` - Process configuration
- `runtime.txt` - Python version
- `requirements.txt` - Dependencies with gunicorn
- `.gitignore` - Ignore unnecessary files

### 2. Push to GitHub

```bash
git add .
git commit -m "Add Render deployment configuration"
git push origin main
```

### 3. Deploy on Render.com

1. **Go to Render.com**
   - Visit: https://render.com/
   - Sign up or login (you can use GitHub)

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository: `safiullah-foragy/Chat_Bot`
   - Render will automatically detect `render.yaml`

3. **Configure (Auto-detected from render.yaml)**
   - Name: `object-detection-api`
   - Runtime: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn -w 2 -b 0.0.0.0:$PORT api_lite:app --timeout 120`
   - Plan: Free (or choose paid for better performance)

4. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes for first build)
   - Model weights will download on first start (~160MB)

### 4. Your API URL

After deployment, you'll get a URL like:
```
https://object-detection-api-xxxx.onrender.com
```

Test it:
```bash
curl https://your-app-name.onrender.com/health
```

## 📝 Important Notes

### Free Tier Limitations
- ⏰ **Spins down after 15 minutes of inactivity**
- 🐌 **First request after sleep takes 50+ seconds to wake up**
- 💾 512 MB RAM (sufficient for CPU inference)
- 🔄 Automatic deploys on git push

### Solutions for Cold Starts
1. **Keep-alive service** (ping every 10 minutes)
2. **Upgrade to paid plan** ($7/month - no sleep)
3. **Use UptimeRobot** to ping your API regularly

### Performance Tips
- First detection: ~5-10 seconds
- Subsequent detections: ~2-3 seconds
- Consider caching if needed

## 🔧 Environment Variables (Optional)

Add in Render Dashboard → Environment:
- `PYTHON_VERSION=3.11.0`
- `MAX_WORKERS=2` (for gunicorn)
- `TIMEOUT=120` (request timeout)

## 📊 Monitor Your API

Render provides:
- Deployment logs
- Runtime logs
- Metrics dashboard
- Health check monitoring

## 🔄 Update Your API

Simply push to GitHub:
```bash
git add .
git commit -m "Update API"
git push origin main
```

Render will automatically rebuild and redeploy!

## ⚠️ Troubleshooting

**Build fails:**
- Check logs in Render dashboard
- Verify requirements.txt is correct
- Ensure Python version is compatible

**Out of memory:**
- Model is large (~160MB)
- Free tier has 512MB RAM limit
- Consider upgrading to paid plan

**Timeout errors:**
- First request after sleep is slow
- Increase timeout in render.yaml
- Use keep-alive service

**Model download slow:**
- First deployment downloads ~160MB
- Cached for future deploys
- Be patient (5-10 minutes)

## 💰 Cost

**Free Tier:**
- 750 hours/month free
- Multiple services allowed
- Perfect for testing

**Paid Tier ($7/month):**
- No sleep/spin down
- Better performance
- More RAM options
- Custom domains

## 🎯 Next Steps

After deployment:
1. Test all endpoints
2. Update Flutter app with new URL
3. Set up monitoring
4. Consider keep-alive solution
5. Monitor usage and performance
