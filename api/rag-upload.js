import { uploadDocument } from './_rag.js';

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  try {
    const { title, content } = req.body || {};
    const result = await uploadDocument({ title, content });
    res.status(200).json({ ok: true, ...result });
  } catch (err) {
    res.status(400).json({ error: err.message || 'Upload failed' });
  }
}
