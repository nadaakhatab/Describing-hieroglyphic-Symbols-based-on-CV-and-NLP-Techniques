"""FastAPI route and small same-origin browser interface for chat."""

import httpx
from fastapi import APIRouter, HTTPException

from .config import Settings
from .knowledge import KnowledgeBase
from .schemas import ChatRequest, ChatResponse
from .service import ChatError, ChatService


def create_chat_router(settings: Settings | None = None, knowledge: KnowledgeBase | None = None) -> APIRouter:
    settings = settings or Settings.from_env()
    knowledge = knowledge if knowledge is not None else KnowledgeBase.load(settings.data_path)
    router = APIRouter(tags=["Chatbot"])

    @router.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
            try:
                return await ChatService(settings, knowledge, client).reply(request)
            except ChatError as exc:
                raise HTTPException(status_code=exc.status, detail=exc.message) from exc

    return router


def chat_demo_html() -> str:
    return """<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Hieroglyph Chatbot</title><style>body{font:16px system-ui;background:#faf7ef;color:#28251e;margin:0;padding:24px}main{max-width:750px;margin:auto}#chat{min-height:220px;background:white;border:1px solid #d8ceb8;border-radius:12px;padding:18px;margin:20px 0}.turn{white-space:pre-wrap;overflow-wrap:anywhere;margin:12px 0;padding:12px;border-radius:8px;background:#f4f0e5}.user{background:#e9f1f1}label{display:block;margin:12px 0 5px}input,textarea,button{font:inherit;box-sizing:border-box;border:1px solid #b7ab90;border-radius:7px;padding:10px}input,textarea{width:100%}textarea{height:90px}button{background:#315553;color:white;cursor:pointer;margin-top:12px}button:disabled{opacity:.5}#status{min-height:24px;color:#8a3028}.note{font-size:14px;color:#68604f}</style><main><h1>Hieroglyph Chatbot</h1><p>Ask about a symbol, Gardiner code, or Ancient Egyptian writing. اسأل بالعربي أو بالإنجليزي.</p><p class=note>This is a text chatbot. It explains supplied glyph codes; it does not inspect images or generate audio.</p><div id=chat aria-live=polite></div><form id=form><label for=codes>Glyph codes (optional, separated by commas)</label><input id=codes placeholder=\"A1, S34\" maxlength=250><label for=message>Your question</label><textarea id=message dir=auto placeholder=\"What does S34 mean?\" maxlength=2000 required></textarea><button id=send>Send</button> <button id=clear type=button>New conversation</button></form><p id=status role=status></p><p class=note>Recent messages and matching project descriptions are sent to Groq. History stays in this page's memory and resets when you reload.</p></main><script>const history=[],chat=document.getElementById('chat'),status=document.getElementById('status'),send=document.getElementById('send'),clear=document.getElementById('clear');function addTurn(role,text){const item=document.createElement('div');item.className='turn '+role;item.dir='auto';item.textContent=(role==='user'?'You: ':'Assistant: ')+text;chat.appendChild(item)}clear.onclick=()=>{history.length=0;chat.replaceChildren();status.textContent=''};document.getElementById('form').onsubmit=async event=>{event.preventDefault();const box=document.getElementById('message'),message=box.value.trim();if(!message)return;const codes=document.getElementById('codes').value.split(',').map(x=>x.trim()).filter(Boolean);if(codes.length>6){status.textContent='Use at most six codes.';return}send.disabled=clear.disabled=true;status.textContent='Thinking…';try{const result=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message,history:history.slice(-10),glyph_codes:codes})}),data=await result.json();if(!result.ok)throw Error(typeof data.detail==='string'?data.detail:'Check your message and codes.');addTurn('user',message);addTurn('assistant',data.response);history.push({role:'user',content:message},{role:'assistant',content:data.response.slice(0,2000)});history.splice(0,Math.max(0,history.length-10));box.value='';status.textContent=data.context_keys.length?'Local context supplied: '+data.context_keys.join(', '):'No matching local description supplied.'}catch(error){status.textContent=error.message}finally{send.disabled=clear.disabled=false}};</script></html>"""
