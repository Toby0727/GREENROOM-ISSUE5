import { answerWithRag } from './_rag.js';

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  try {
    const { messages = [] } = req.body || {};
    const cleanMessages = Array.isArray(messages) ? messages : [];
    const lastUser = [...cleanMessages].reverse().find(m => m?.role === 'user');

    if (!lastUser?.content) {
      return res.status(400).json({ error: 'A user message is required' });
    }

    const result = await answerWithRag({
      message: String(lastUser.content),
      history: cleanMessages,
      sessionId: 'default'
    });

    res.status(200).json({ reply: result.reply, mode: result.mode, citations: result.citations });
  } catch (err) {
    res.status(500).json({ error: err.message || 'Chat failed' });
  }
}
