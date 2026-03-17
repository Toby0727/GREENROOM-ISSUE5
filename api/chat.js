export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).end();

  const { messages } = req.body;

  try {
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [
          { role: 'system', content: 'You are a sharp, witty assistant displayed on a TV screen. Reply in at most 2 sentences and no more than 3 short lines. Be direct and interesting.' },
          ...messages
        ],
        max_tokens: 150
      })
    });

    const data = await response.json();
    if (!response.ok) return res.status(response.status).json(data);

    const rawReply = data?.choices?.[0]?.message?.content || '';
    const reply = limitReply(rawReply);
    res.status(200).json({ reply });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
}

function limitReply(text) {
  const compact = String(text || '').replace(/\r\n/g, '\n').trim();

  // Keep at most 2 sentence-like chunks.
  const sentenceParts = compact
    .split(/(?<=[.!?])\s+/)
    .map(s => s.trim())
    .filter(Boolean)
    .slice(0, 2);

  let out = sentenceParts.join(' ');
  if (!out) out = compact;

  // Ensure no more than 3 lines.
  const lines = out
    .split('\n')
    .map(l => l.trim())
    .filter(Boolean)
    .slice(0, 3);

  return lines.join('\n');
}
