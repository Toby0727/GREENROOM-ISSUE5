# GREENROOM-ISSUE5

## Durable Comments Storage

Comments now use Vercel KV when these environment variables are present:

- `KV_REST_API_URL`
- `KV_REST_API_TOKEN`

Without those variables, the app falls back to local file storage for development only.

### Vercel Setup

1. Create a Vercel KV store and attach it to this project.
2. Ensure `KV_REST_API_URL` and `KV_REST_API_TOKEN` are available to the deployment.
3. Redeploy the project.