import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import './App.css'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = async () => {
    const query = input.trim()
    if (!query || loading) return

    setMessages(prev => [...prev, { role: 'user', text: query }])
    setInput('')
    setLoading(true)

    // Add empty AI message placeholder — tokens will fill it in
    setMessages(prev => [...prev, { role: 'ai', text: '', intent: null }])

    try {
      const res = await fetch('http://localhost:8000/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      })

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const data = JSON.parse(line.slice(6))

          if (data.token !== undefined) {
            setMessages(prev => {
              const updated = [...prev]
              const last = updated[updated.length - 1]
              updated[updated.length - 1] = { ...last, text: last.text + data.token }
              return updated
            })
          }

          if (data.done) {
            setMessages(prev => {
              const updated = [...prev]
              updated[updated.length - 1] = { ...updated[updated.length - 1], intent: data.intent }
              return updated
            })
          }

          if (data.error) {
            setMessages(prev => {
              const updated = [...prev]
              updated[updated.length - 1] = { ...updated[updated.length - 1], text: 'Something went wrong.' }
              return updated
            })
          }
        }
      }
    } catch (err) {
      setMessages(prev => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last?.role === 'ai') {
          updated[updated.length - 1] = { ...last, text: 'Something went wrong. Make sure the backend is running.' }
        }
        return updated
      })
    } finally {
      setLoading(false)
    }
  }

  const clearConversation = async () => {
    setMessages([])
    await fetch('http://localhost:8000/clear', { method: 'POST' })
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="app">
      <div className="header">
        <h1>NCERT AI Tutor</h1>
        <button className="clear-btn" onClick={clearConversation}>
          New Chat
        </button>
      </div>

      <div className="messages">
        {messages.length === 0 && (
          <div style={{ textAlign: 'center', color: '#444', marginTop: '40px' }}>
            <p>Ask anything from your NCERT Science textbook</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            {msg.role === 'ai' && msg.intent && (
              <div className="intent-badge">{msg.intent}</div>
            )}
            {msg.role === 'ai'
              ? <ReactMarkdown>{msg.text}</ReactMarkdown>
              : msg.text
            }
          </div>
        ))}

        {loading && (
          <div className="loading">Thinking...</div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="input-bar">
        <input
          type="text"
          placeholder="Ask a question from your NCERT textbook..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
        />
        <button onClick={sendMessage} disabled={loading}>
          {loading ? '...' : 'Ask'}
        </button>
      </div>
    </div>
  )
}

export default App
