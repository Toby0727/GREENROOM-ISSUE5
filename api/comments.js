import { clearComments, readComments, writeComments } from './_comments.js';

function setCorsHeaders(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
}

export default async function handler(req, res) {
  setCorsHeaders(res);

  if (req.method === 'OPTIONS') return res.status(200).end();

  try {
    if (req.method === 'GET') {
      const comments = await readComments();
      return res.status(200).json({ comments });
    }

    if (req.method === 'POST') {
      const comments = await writeComments(req.body?.comments);
      return res.status(200).json({ comments });
    }

    if (req.method === 'DELETE') {
      await clearComments();
      return res.status(200).json({ comments: [] });
    }

    return res.status(405).json({ error: 'Method not allowed' });
  } catch (err) {
    return res.status(500).json({ error: err.message || 'Comments request failed' });
  }
}