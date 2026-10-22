const $=s=>document.querySelector(s);let docs=[],cur=null,curChat=[],page=1,hl=null,poll=null;
const api=async(u,o)=>{const r=await fetch(u,o);const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.error||r.statusText);return j};
const el=(t,c,x)=>{const e=document.createElement(t);if(c)e.className=c;if(x!==undefined)e.textContent=x;return e};
let pending=null;
async function loadDocs(){docs=await api('/api/docs');renderDocs();const pd=pending&&docs.find(d=>d.id===pending);if(pd&&pd.status!=='indexing'){pending=null;if(pd.status==='ready')return selectDoc(pd.id)}if(docs.some(d=>d.status==='indexing')){clearTimeout(poll);poll=setTimeout(loadDocs,1500)}
 if(cur){const d=docs.find(d=>d.id===cur);if(!d){cur=null;showDoc(null)}else if(d.status==='ready'&&$('#q').disabled)selectDoc(cur)}
 else if(!pending){const r=docs.filter(d=>d.status==='ready').pop();if(r)selectDoc(r.id)}}
function renderDocs(){const box=$('#docs');box.innerHTML='';docs.slice().reverse().forEach(d=>{const e=el('div','doc'+(d.id===cur?' on':'')+(d.status==='error'?' err':''));e.append(el('b',0,d.name));
 const s=d.status==='ready'?`${d.pages} pages · ${d.chunks} passages · indexed in ${d.seconds} s`+(d.skipped&&d.skipped.length?` · ${d.skipped.length} without text`:''):d.status==='indexing'?'reading and indexing…':d.error;e.append(el('small',0,s));
 if(d.status==='indexing'){const b=el('div','bar');b.append(el('i'));e.append(b)}
 const x=el('button','del','delete');x.onclick=async ev=>{ev.stopPropagation();if(confirm(`Delete "${d.name}" from the library?`)){await api('/api/doc/'+d.id,{method:'DELETE'});if(cur===d.id){cur=null;showDoc(null)}loadDocs()}};e.append(x);
 if(d.status==='ready')e.onclick=()=>selectDoc(d.id);box.append(e)})}
function showDoc(d){const has=!!d;$('#empty').hidden=has;$('#sheet').hidden=!has;$('#nav').hidden=!has;$('#q').disabled=!has;$('#log').innerHTML='';if(!has)hint(null)}
async function selectDoc(id){cur=id;const d=await api('/api/doc/'+id);showDoc(d);renderDocs();curChat=d.chat||[];$('#log').innerHTML='';
 if(!curChat.length)hint(d);curChat.forEach((r,i)=>addTurn(r,i+1,false));const L=curChat[curChat.length-1];if(L&&L.highlight)go(L.highlight.page,L.highlight.rects);else go(1,null);$('#q').focus();window.curMeta=d}
function hint(d){const h=el('div','hint');h.id='hint';if(!d){h.textContent='Your questions and answers appear here, each with the page it came from.'}else{h.append('Ask a question about ');h.append(el('b',0,d.name));h.append('. Try:');
 ['What is this document about?','Summarise the main points on page 1','Who is listed as the author or issuer?'].forEach(t=>{const x=el('div',0,t);x.onclick=()=>{$('#q').value=t;$('#form').requestSubmit()};h.append(x)})}$('#log').append(h)}
function go(n,rects){const d=docs.find(d=>d.id===cur);if(!d)return;n=Math.max(1,Math.min(d.pages,n));page=n;const im=$('#pageimg');im.onload=()=>drawHl(rects);im.src=`/api/doc/${cur}/page/${n}.png`;$('#pn').textContent=`page ${n} of ${d.pages}`;hl=rects;drawHl(rects);$('#view').scrollTop=0}
function drawHl(rects){const h=$('#hls');h.innerHTML='';(rects||[]).forEach(r=>{const e=el('div','hl');const p=0.004;e.style.left=(r[0]-p)*100+'%';e.style.top=(r[1]-p/2)*100+'%';e.style.width=(r[2]-r[0]+2*p)*100+'%';e.style.height=(r[3]-r[1]+p)*100+'%';h.append(e)})}
$('#prev').onclick=()=>go(page-1,null);$('#next').onclick=()=>go(page+1,null);
document.addEventListener('keydown',e=>{if(document.activeElement.id==='q')return;if(e.key==='ArrowLeft')go(page-1,null);if(e.key==='ArrowRight')go(page+1,null)});
function addTurn(r,n,scroll=true){const log=$('#log');$('#hint')&&$('#hint').remove();const t=el('div');t.append(el('div','n','QUESTION '+n),el('div','q',r.question));
 if(r.error){t.append(el('div','warn',r.error))}else{
 const st=el('div','steps');['clean','hybrid','rerank','answer'].forEach(s=>st.append(el('i',0,s)));const tm=r.timing||{};st.append(((tm.hybrid||0)+(tm.rerank||0)+(tm.answer||0)).toFixed(1)+' s');t.append(st);
 const a=el('div','a');if(r.answer===null){const x=r.sources[0];a.append(el('div',0,'Best passage, page '+x.page+': '+x.text.slice(0,260)+(x.text.length>260?'…':'')+' '));if(!window.noteShown){window.noteShown=1;a.append(el('div','warn','No GROQ_API_KEY set, so no written answer. Add it to .env and restart.'))}}
 else if(r.answer.toUpperCase().startsWith('NOT FOUND')||!r.answer){a.textContent='This document does not seem to contain the answer.'}else a.append(r.answer+' ');
 (r.cited||[]).forEach((p,i)=>{const c=el('button','cite','p.'+p);c.onclick=()=>go(p,i===0&&r.highlight&&r.highlight.page===p?r.highlight.rects:null);a.append(c)});t.append(a);
 const s=el('div','src');const dt=el('details');dt.append(el('summary',0,(r.cited&&r.cited.length?'Read from page '+r.cited.join(', ')+(r.highlight&&r.highlight.rects?' · click the page chip to see the line':''):'Closest passages')+' · show the 3 passages used'));
 (r.sources||[]).forEach(x=>{const p=el('p',0,`p.${x.page}  ${x.text.slice(0,320)}${x.text.length>320?'…':''}`);dt.append(p)});s.append(dt);t.append(s);
 if(r.highlight&&scroll)go(r.highlight.page,r.highlight.rects)}
 log.append(t);if(scroll)log.scrollTop=log.scrollHeight}
$('#form').onsubmit=async e=>{e.preventDefault();const q=$('#q').value.trim();if(!q||!cur)return;$('#q').value='';$('#q').disabled=true;
 const log=$('#log');$('#hint')&&$('#hint').remove();const w=el('div');w.append(el('div','n','QUESTION '+(curChat.length+1)),el('div','q',q),el('div','steps','reading the document…'));log.append(w);log.scrollTop=log.scrollHeight;
 try{const r=await api(`/api/doc/${cur}/ask`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({q})});w.remove();curChat.push(r);addTurn(r,curChat.length)}
 catch(err){w.remove();addTurn({question:q,error:err.message},curChat.length+1)}$('#q').disabled=false;$('#q').focus()};
async function upload(f){if(!f)return;if(f.type!=='application/pdf'&&!f.name.toLowerCase().endsWith('.pdf')){alert('Please choose a PDF file.');return}
 const fd=new FormData();fd.append('file',f);try{const r=await api('/api/upload',{method:'POST',body:fd});pending=r.id;await loadDocs()}catch(e){alert(e.message)}}
$('#file').onchange=e=>{upload(e.target.files[0]);e.target.value=''};
['dragenter','dragover'].forEach(t=>document.addEventListener(t,e=>{e.preventDefault();document.body.classList.add('dragging')}));
['dragleave','drop'].forEach(t=>document.addEventListener(t,e=>{e.preventDefault();if(t==='drop'||e.target===document.documentElement)document.body.classList.remove('dragging')}));
document.addEventListener('drop',e=>{const f=e.dataTransfer.files[0];upload(f)});
$('#drop').onclick=()=>$('#file').click();
api('/api/status').then(s=>{if(!s.has_key)$('#keynote').hidden=false});
showDoc(null);loadDocs();
