export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  try {
    const { answerWithRag } = await import('./_rag.js');
    const { message, history = [], sessionId = 'default' } = req.body || {};
    if (!message || !String(message).trim()) {
      return res.status(400).json({ error: 'message is required' });
    }

    const result = await answerWithRag({
      message: String(message).trim(),
      history: Array.isArray(history) ? history : [],
      sessionId: String(sessionId || 'default')
    });

    res.status(200).json(result);
  } catch (err) {
    res.status(500).json({ error: err.message || 'RAG chat failed' });
  }
}
