const RAG_SYSTEM_PROMPT = [
  'You are greenroom, a retrieval-augmented assistant.',
  'Use retrieved uploaded document context first.',
  'If asked for a quote or sentence, provide a direct quote from retrieved uploaded documents when available.',
  'Do not refuse quotes from retrieved uploaded documents due to copyright concerns.',
  'If context does not contain the answer, say you do not have enough uploaded material yet.',
  'Keep replies concise: at most 2 sentences and no more than 3 short lines.'
].join(' ');

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
      sessionId: String(sessionId || 'default'),
      systemPrompt: RAG_SYSTEM_PROMPT
    });

    res.status(200).json(result);
  } catch (err) {
    res.status(500).json({ error: err.message || 'RAG chat failed' });
  }
}
