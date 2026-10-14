// Ко всем изменяющим запросам автоматически добавляем CSRF-токен —
// так его не нужно проставлять в каждом вызове fetch по отдельности.
const CSRF = document.querySelector('meta[name="csrf-token"]').content;
(function(){
  const orig = window.fetch;
  window.fetch = function(url, opts){
    opts = opts || {};
    const m = (opts.method || 'GET').toUpperCase();
    if(m !== 'GET' && m !== 'HEAD'){
      opts.headers = new Headers(opts.headers || {});
      opts.headers.set('X-CSRF-Token', CSRF);
      opts.credentials = opts.credentials || 'same-origin';
    }
    return orig.call(this, url, opts);
  };
})();
const $ = s => document.querySelector(s);
const logEl = $('#log');

// --- предпросмотр шрифта субтитров ---
const fontSel = $('#fontSel'), fontPrev = $('#fontPrev');
function syncFont(){
  const v = fontSel.value || 'Montserrat';
  // Montserrat может не быть в системе — подставляем разумный запасной стек
  fontPrev.style.fontFamily = `"${v}", "Segoe UI", Arial, sans-serif`;
}
fontSel.addEventListener('change', syncFont); syncFont();

// --- живой предпросмотр цветов субтитров и хука ---
const subColor = $('#subColor'), subHl = $('#subHl'), nohook = $('#nohook');
function syncColors(){
  const base = subColor.value, hl = subHl.value;
  document.querySelectorAll('#fontPrev .fp-w').forEach(e => e.style.color = base);
  $('#fpHl').style.color = hl;
  $('#fpHook').style.color = base;
  $('#subColorTxt').textContent = base.replace('#','').toUpperCase();
  $('#subHlTxt').textContent = hl.replace('#','').toUpperCase();
  $('#fpHook').style.display = nohook.checked ? 'none' : 'block';
}
[subColor, subHl, nohook].forEach(el => el.addEventListener('input', syncColors));
syncColors();

// --- визуализация силы обрезки ---
const cropRange = $('#cropRange'), cropNum = $('#cropNum');
function syncCrop(from){
  let v = parseFloat((from === 'num' ? cropNum.value : cropRange.value));
  if(isNaN(v)) v = 0.15;
  v = Math.min(0.35, Math.max(0, v));
  cropRange.value = v; cropNum.value = v.toFixed(2);

  const keepPct = (1 - 2*v) * 100;                 // сколько ширины остаётся
  $('#cutL').style.width = (v*100) + '%';
  $('#cutR').style.width = (v*100) + '%';
  $('#cvKeep').style.width = keepPct + '%';
  $('#cvKeepTxt').textContent = Math.round(keepPct) + '%';

  // та же математика, что в _render_wide: вписываем оставшуюся ширину в 1080,
  // высота чёткой полосы = 1080 * (src_h / kept_w), src 16:9 -> src_h/src_w=9/16
  const fgH = Math.min(1, (1080 * (9/16) / (1 - 2*v)) / 1920);
  $('#cvFg').style.height = (fgH*100) + '%';
  $('#cvFgTxt').textContent = Math.round(fgH*100) + '%';

  const zoom = 1/(1 - 2*v);
  $('#cropNote').textContent = v === 0
    ? 'Кадр целиком, ничего не обрезается — люди самые мелкие, полосы самые толстые.'
    : `Отрезается по ${Math.round(v*100)}% с каждой стороны (останется ${Math.round(keepPct)}% ширины). `
      + `Люди крупнее в ${zoom.toFixed(2)}× — чёткая часть займёт ${Math.round(fgH*100)}% высоты экрана.`;
}
cropRange.addEventListener('input', ()=>syncCrop('range'));
cropNum.addEventListener('input', ()=>syncCrop('num'));
syncCrop('range');

// подсветка выбранного режима + показ поля обрезки
function syncModes(){
  document.querySelectorAll('#modes label').forEach(l=>{
    const r = l.querySelector('input');
    l.classList.toggle('on', r.checked);
  });
  const st = document.querySelector('input[value=static]').checked;
  $('#cropBox').style.display = st ? 'block' : 'none';
}
document.querySelectorAll('input[name=camera_mode]').forEach(r=>r.addEventListener('change',syncModes));
syncModes();

// --- задачи ---
// Обработка идёт в отдельном процессе-воркере, поэтому мы следим за
// конкретной задачей по её идентификатору, а не за «состоянием сервера».
let _jobId = null, _es = null;

function setRunning(on, text){
  $('#go').disabled = on;
  $('#stopBtn').style.display = on ? 'inline-block' : 'none';
  $('#dot').classList.toggle('on', on);
  $('#stat').textContent = text || (on ? 'идёт обработка…' : 'простой');
}

function watchJob(id){
  if(_es) _es.close();
  _jobId = id;
  logEl.textContent = '';
  showProgress({percent:0, label:'Ставим в очередь…', detail:'', eta:''});
  setRunning(true, 'задача поставлена в очередь…');
  _es = new EventSource('/api/jobs/' + id + '/log');
  _es.onmessage = e => {
    const line = JSON.parse(e.data);
    if(!line) return;
    logEl.textContent += line;
    logEl.scrollTop = logEl.scrollHeight;
  };
  _es.addEventListener('end', e => {
    _es.close(); _es = null; _jobId = null;
    const st = JSON.parse(e.data);
    if(st === 'done'){
      showProgress({percent:100, label:'Готово', detail:'Клипы ниже', eta:''});
      setTimeout(()=>hideProgress('Готово. Можно запускать следующее видео.'), 4000);
    } else {
      hideProgress(st === 'canceled' ? 'Обработка отменена.'
                                     : 'Обработка завершилась ошибкой — см. лог.');
    }
    setRunning(false, st === 'done' ? 'готово' :
                      st === 'canceled' ? 'отменено' : 'завершено с ошибкой');
    loadResults(); // --- список моих задач ---
const JOB_ST = {done:'готово', running:'идёт', queued:'в очереди',
                error:'ошибка', canceled:'отменена'};

function fmtWhen(iso){
  if(!iso) return '';
  const d = new Date(iso), now = new Date();
  const mins = Math.round((now - d) / 60000);
  if(mins < 1) return 'только что';
  if(mins < 60) return mins + ' мин назад';
  if(mins < 24*60) return Math.round(mins/60) + ' ч назад';
  return d.toLocaleDateString('ru-RU', {day:'numeric', month:'short'});
}

async function loadJobs(){
  const box = $('#jobsList');
  try{
    const list = await (await fetch('/api/jobs')).json();
    if(!Array.isArray(list) || !list.length){
      box.innerHTML = '<div class="noRes">Задач пока нет.</div>';
      return;
    }
    box.innerHTML = list.slice(0, 20).map(j => {
      const p = j.progress_info || {};
      const st = j.status || 'queued';
      const meta = st === 'running' || st === 'queued'
        ? escapeHtml(p.label || '') + (p.eta ? ' · осталось ' + escapeHtml(p.eta) : '')
        : (st === 'done' ? (j.clips_count || 0) + ' клипов' : escapeHtml(j.error || ''));
      const canStop = (st === 'queued' || st === 'running');
      return `<div class="jobRow">
        <div class="jobMain">
          <div class="jobTitle">${escapeHtml(j.title || 'без названия')}</div>
          <div class="jobMeta">${fmtWhen(j.created_at)}${meta ? ' · ' + meta : ''}</div>
        </div>
        <span class="jobSt st-${st}">${JOB_ST[st] || st}</span>
        ${canStop ? `<button class="ghost" data-act="stopJob" data-arg="${j.id}">Отменить</button>` : ''}
      </div>`;
    }).join('');
  }catch(e){
    box.innerHTML = '<div class="noRes">Не удалось загрузить список.</div>';
  }
}

async function toggleNotify(cb){
  try{
    await fetch('/api/notify', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({enabled: cb.checked})});
  }catch(e){ cb.checked = !cb.checked; }
}

async function stopJob(id){
  await fetch('/api/jobs/' + id + '/cancel', {method:'POST'});
  loadJobs();
}

// Обновляем список, пока есть незавершённые задачи: реже, чем прогресс,
// чтобы не нагружать сервер лишними запросами.
setInterval(() => { if(document.visibilityState === 'visible') loadJobs(); }, 20000);
loadJobs();

loadMe();
  });
  _es.onerror = () => { setRunning(false, 'связь потеряна — обновите страницу'); };
  pollStatus();
}

// Показываем, ЧТО именно происходит. Ожидание переносится плохо не из-за
// длительности, а из-за неизвестности — поэтому опрашиваем состояние часто
// и показываем реальный этап, а не полоску по таймеру.
function showProgress(p){
  if(!p) return;
  $('#progWrap').style.display = 'block';
  $('#progIdle').style.display = 'none';
  $('#progLabel').textContent = p.label || '';
  $('#progFill').style.width = (p.percent || 0) + '%';
  $('#progDetail').textContent = p.detail || '';
  $('#progEta').textContent = p.eta ? ('осталось ' + p.eta) : '';
}

function hideProgress(text){
  $('#progWrap').style.display = 'none';
  $('#progIdle').style.display = 'block';
  $('#progIdle').textContent = text || 'Заполните форму слева и нажмите «Запустить».';
}

async function pollStatus(){
  if(!_jobId) return;
  try{
    const j = await (await fetch('/api/jobs/' + _jobId)).json();
    showProgress(j.progress_info);
    if(j.status === 'queued' || j.status === 'running'){
      setRunning(true, j.progress_info ? j.progress_info.label : 'обработка…');
      // В очереди опрашиваем чаще: позиция меняется, и человеку важно видеть
      // движение. В работе прогресс идёт медленнее — реже.
      setTimeout(pollStatus, j.status === 'queued' ? 4000 : 6000);
    }
  }catch(e){ setTimeout(pollStatus, 8000); }
}

$('#f').addEventListener('submit', async ev => {
  ev.preventDefault();
  $('#err').style.display='none';
  const fd = new FormData($('#f'));
  setRunning(true, 'отправляю…');
  const r = await fetch('/api/start',{method:'POST',body:fd});
  const j = await r.json();
  if(!j.ok){
    setRunning(false);
    $('#err').textContent = j.error; $('#err').style.display='block';
    return;
  }
  watchJob(j.job_id);
  loadJobs();
});

$('#stopBtn').onclick = async ()=>{
  if(!_jobId) return setRunning(false);
  await fetch('/api/jobs/' + _jobId + '/cancel', {method:'POST'});
  setRunning(false, 'отменяю…');
};
$('#clearBtn').onclick = ()=>{ logEl.textContent=''; };
$('#refresh').onclick = loadResults;

let _items = [];

function viralClass(v){
  if(v === null || v === undefined) return 'na';
  if(v >= 8) return 'hi';
  if(v >= 6.5) return 'mid';
  return 'low';
}

function renderResults(){
  const box = $('#res');
  const sort = $('#sortSel').value;
  const folder = $('#folderSel').value;
  let list = _items.filter(i => !folder || i.folder === folder);

  const cmp = {
    // клипы без виральности (нет манифеста) уходят в конец
    virality: (a,b) => (b.virality ?? -1) - (a.virality ?? -1),
    new:      (a,b) => b.mtime - a.mtime,
    old:      (a,b) => a.mtime - b.mtime,
    name:     (a,b) => a.name.localeCompare(b.name, 'ru'),
    size:     (a,b) => b.size_mb - a.size_mb,
    duration: (a,b) => (b.duration ?? -1) - (a.duration ?? -1),
  }[sort];
  list = list.slice().sort(cmp);

  if(!list.length){
    box.innerHTML = '<div style="color:var(--muted);font-size:13px">Пока пусто.</div>';
    return;
  }
  box.innerHTML = list.map(i => {
    const v = (i.virality ?? null);
    const vTxt = v === null ? '—' : v.toFixed(1);
    const chips = [];
    if(i.duration) chips.push(`${i.duration} с`);
    if(i.parts && i.parts > 1) chips.push(`${i.parts} сегм.`);
    chips.push(`${i.size_mb} МБ`);
    return `
    <div class="item">
      <span class="vir ${viralClass(v)}" title="Оценка виральности (1–10)">${vTxt}</span>
      <button type="button" class="del" title="Удалить клип"
              data-act="deleteClip" data-arg="${escapeHtml(i.rel)}">✕</button>
      <video src="/clips/${encodeURI(i.rel)}" controls preload="metadata"></video>
      ${i.title ? `<div class="ttl">${escapeHtml(i.title)}</div>` : ''}
      <div class="meta">${chips.map(c=>`<span class="chip">${c}</span>`).join('')}</div>
      <div class="nm">${escapeHtml(i.name)}<br>${escapeHtml(i.folder)}</div>
    </div>`;
  }).join('');
}

function escapeHtml(s){
  return String(s).replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function loadResults(){
  const r = await fetch('/api/results');
  _items = await r.json();
  // обновляем список папок, сохраняя выбор
  const fs = $('#folderSel'), cur = fs.value;
  const folders = [...new Set(_items.map(i=>i.folder))].sort();
  fs.innerHTML = '<option value="">Все папки</option>' +
    folders.map(f=>`<option value="${escapeHtml(f)}">${escapeHtml(f)}</option>`).join('');
  if(folders.includes(cur)) fs.value = cur;
  renderResults();
}

$('#sortSel').addEventListener('change', renderResults);
$('#folderSel').addEventListener('change', renderResults);

// --- удаление клипа ---
async function deleteClip(rel){
  const name = decodeURI(rel).split('/').pop();
  if(!confirm(`Удалить клип «${name}»?\n\nФайл будет удалён с диска безвозвратно.`)) return;
  const r = await fetch('/api/delete', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({rel: decodeURI(rel)})});
  const j = await r.json();
  if(!j.ok && j.errors && j.errors.length) alert('Не удалось удалить:\n' + j.errors.join('\n'));
  loadResults();
}

// --- профиль и плашка остатка доступа ---
async function loadMe(){
  const r = await fetch('/api/me');
  if(r.status === 401 || r.status === 403){ location.href='/login'; return; }
  const me = await r.json();
  $('#whoami').textContent = me.email;
  if(me.is_admin) $('#adminBtn').style.display = 'inline-block';

  // Переключатель писем показываем только если почта на сервере настроена —
  // иначе он обещал бы то, чего не будет.
  const nb = $('#notifyBox');
  if(nb){
    nb.style.display = me.notify_available ? 'block' : 'none';
    const cb = $('#notifyCb');
    if(cb) cb.checked = !!me.notify;
  }

  const pill = $('#daysPill'), txt = $('#daysTxt');
  if(me.days_left === null || me.days_left === undefined){
    txt.textContent = 'без ограничения'; pill.className = 'pill ok';
  } else if(me.days_left <= 0){
    const h = me.hours_left ?? 0;
    txt.textContent = h > 0 ? `${h} ч осталось` : 'доступ истёк';
    pill.className = 'pill crit';
  } else {
    txt.textContent = `${me.days_left} ${plural(me.days_left)} осталось`;
    pill.className = 'pill ' + (me.days_left <= 3 ? 'crit'
                    : me.days_left <= 7 ? 'warn' : 'ok');
  }
}
function plural(n){
  const a = Math.abs(n) % 100, b = a % 10;
  if(a > 10 && a < 20) return 'дней';
  if(b > 1 && b < 5) return 'дня';
  if(b === 1) return 'день';
  return 'дней';
}

$('#logoutBtn').onclick = async () => {
  await fetch('/api/logout', {method:'POST'});
  location.href = '/login';
};

// --- админка: управление пользователями ---
const modal = $('#adminModal');
$('#adminBtn').onclick = () => { modal.classList.add('on'); showTab('users'); };
$('#admClose').onclick = () => modal.classList.remove('on');
modal.addEventListener('click', e => { if(e.target === modal) modal.classList.remove('on'); });

$('#genPass').onclick = () => {
  const cs = 'abcdefghjkmnpqrstuvwxyz23456789';
  $('#nuPass').value = Array.from({length:10}, () =>
    cs[Math.floor(Math.random()*cs.length)]).join('');
};

async function adminPost(url, body){
  const r = await fetch(url, {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  return r.json();
}

$('#nuCreate').onclick = async () => {
  const e = $('#admErr'); e.style.display='none';
  const j = await adminPost('/api/admin/users', {
    email: $('#nuEmail').value, password: $('#nuPass').value,
    days: parseInt($('#nuDays').value || '30', 10), note: $('#nuNote').value});
  if(!j.ok){ e.textContent = j.error; e.style.display='block'; return; }
  alert(`Аккаунт создан.\n\nПочта: ${$('#nuEmail').value}\nПароль: ${$('#nuPass').value}\n\n`
        + 'Передайте эти данные пользователю — пароль больше нигде не показывается.');
  $('#nuEmail').value=''; $('#nuPass').value=''; $('#nuNote').value='';
  loadUsers();
};

// --- вкладки админки: пользователи / расходы ---
function showTab(which){
  const isCost = which === 'cost';
  $('#paneCost').style.display  = isCost ? 'block' : 'none';
  $('#paneUsers').style.display = isCost ? 'none'  : 'block';
  $('#tabCost').classList.toggle('on', isCost);
  $('#tabUsers').classList.toggle('on', !isCost);
  if(isCost) loadUsage(); else loadUsers();
}
$('#tabUsers').onclick = () => showTab('users');
$('#tabCost').onclick  = () => showTab('cost');

function usd(v){ return '$' + (v < 0.01 ? v.toFixed(5) : v.toFixed(4)); }
function nfmt(n){ return (n||0).toLocaleString('ru-RU'); }

async function loadUsage(){
  const r = await fetch('/api/admin/usage');
  const u = await r.json();
  if(u.error){ $('#costCards').innerHTML = `<div class="noRes">${escapeHtml(u.error)}</div>`; return; }

  $('#costCards').innerHTML = `
    <div class="costCard hl"><div class="k">Себестоимость видео</div>
      <div class="v">${usd(u.avg_cost_per_run)}</div>
      <div class="s">в среднем за прогон (только LLM)</div></div>
    <div class="costCard hl"><div class="k">Себестоимость клипа</div>
      <div class="v">${usd(u.avg_cost_per_clip)}</div>
      <div class="s">на один готовый клип</div></div>
    <div class="costCard"><div class="k">Всего потрачено</div>
      <div class="v">${usd(u.total_cost)}</div>
      <div class="s">${nfmt(u.calls)} запросов к ИИ</div></div>
    <div class="costCard"><div class="k">Обработано</div>
      <div class="v">${nfmt(u.runs)}</div>
      <div class="s">видео → ${nfmt(u.clips)} клипов</div></div>
    <div class="costCard"><div class="k">Токенов</div>
      <div class="v">${nfmt(u.total_tok_in + u.total_tok_out)}</div>
      <div class="s">вход ${nfmt(u.total_tok_in)} · выход ${nfmt(u.total_tok_out)}</div></div>
    <div class="costCard"><div class="k">Время прогона</div>
      <div class="v">${Math.round(u.avg_seconds_per_run/60)} мин</div>
      <div class="s">в среднем на видео</div></div>`;

  const st = Object.entries(u.by_stage);
  $('#costStages').innerHTML = !st.length ? '<div class="noRes">Пока нет данных.</div>' : `
    <table class="cost">
      <tr><th>Этап</th><th class="num">Запросов</th><th class="num">Токенов вх.</th>
          <th class="num">Токенов вых.</th><th class="num">Стоимость</th></tr>
      ${st.sort((a,b)=>b[1].cost-a[1].cost).map(([name,s])=>`
        <tr><td>${escapeHtml(name)}</td><td class="num">${nfmt(s.calls)}</td>
            <td class="num">${nfmt(s.tok_in)}</td><td class="num">${nfmt(s.tok_out)}</td>
            <td class="num">${usd(s.cost)}</td></tr>`).join('')}
    </table>`;

  $('#costRuns').innerHTML = !u.recent.length ? '<div class="noRes">Пока нет прогонов.</div>' : `
    <table class="cost">
      <tr><th>Когда</th><th>Источник</th><th class="num">Клипов</th>
          <th class="num">Время</th><th class="num">Стоимость</th></tr>
      ${u.recent.map(r=>`
        <tr><td>${new Date(r.ts).toLocaleString('ru-RU',{day:'2-digit',month:'2-digit',
              hour:'2-digit',minute:'2-digit'})}</td>
            <td style="max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
                title="${escapeHtml(r.source)}">${escapeHtml(r.source)}</td>
            <td class="num">${r.clips}</td>
            <td class="num">${Math.round(r.seconds/60)} мин</td>
            <td class="num">${usd(r.cost)}</td></tr>`).join('')}
    </table>`;

  const p = u.prices[Object.keys(u.prices)[0]];
  $('#costNote').innerHTML =
    'Токены оцениваются по длине текста (погрешность ~10–20%) — для прикидки '
    + 'себестоимости этого достаточно. Цены заданы в clipper/usage.py, обнови их, '
    + 'если тариф провайдера изменится.<br><b>Важно:</b> здесь только расходы на ИИ. '
    + 'Вычисление (Whisper + рендер) сейчас идёт на твоём компьютере бесплатно — '
    + 'при переезде на сервер с видеокартой это станет основной статьёй затрат.';
}

$('#costReset').onclick = async () => {
  if(!confirm('Обнулить всю статистику расходов?')) return;
  await fetch('/api/admin/usage/reset', {method:'POST'});
  loadUsage();
};

let _users = [];            // все аккаунты с сервера
let _selected = new Set();  // выбранные почты

async function loadUsers(){
  const r = await fetch('/api/admin/users');
  _users = await r.json();
  // выбранными оставляем только тех, кто ещё существует
  _selected = new Set([..._selected].filter(e => _users.some(u => u.email === e)));
  renderUsers();
}

function visibleUsers(){
  const q = ($('#admSearch').value || '').trim().toLowerCase();
  const f = $('#admFilter').value;
  return _users.filter(u => {
    if(q && !(u.email.toLowerCase().includes(q) ||
              (u.note||'').toLowerCase().includes(q))) return false;
    if(f === 'active'  && !u.active) return false;
    if(f === 'expired' &&  u.active) return false;
    if(f === 'soon'    && !(u.active && u.days_left !== null && u.days_left <= 7)) return false;
    return true;
  });
}

function renderUsers(){
  const list = visibleUsers();
  $('#admCount').textContent = list.length === _users.length
    ? `всего: ${_users.length}`
    : `найдено: ${list.length} из ${_users.length}`;

  $('#usersList').innerHTML = !list.length
    ? '<div class="noRes">Ничего не найдено.</div>'
    : `<table class="users">
      <tr><th class="chk"></th><th>Почта</th><th>Статус</th><th>Осталось</th>
          <th>Заметка</th><th></th></tr>
      ${list.map(u => `
        <tr class="${_selected.has(u.email) ? 'sel' : ''}">
          <td class="chk"><input type="checkbox" class="uchk" data-email="${escapeHtml(u.email)}"
              ${_selected.has(u.email) ? 'checked' : ''}></td>
          <td>${escapeHtml(u.email)} ${u.is_admin ? '<span class="tag adm">админ</span>' : ''}</td>
          <td><span class="tag ${u.active?'on':'off'}">${u.active?'активен':'истёк'}</span></td>
          <td>${u.days_left === null ? '—' : u.days_left + ' дн.'}</td>
          <td style="color:var(--muted)">${escapeHtml(u.note||'')}</td>
          <td><div class="act">
            <button class="ghost" data-act="extendUser" data-arg="${escapeHtml(u.email)}" data-days="30">+30 дн</button>
            <button class="ghost" data-act="extendUser" data-arg="${escapeHtml(u.email)}" data-days="7">+7</button>
            <button class="ghost" data-act="resetPass" data-arg="${escapeHtml(u.email)}">пароль</button>
            ${u.is_admin ? '' :
              `<button class="stop" data-act="delUser" data-arg="${escapeHtml(u.email)}">удалить</button>`}
          </div></td>
        </tr>`).join('')}
    </table>`;

  document.querySelectorAll('.uchk').forEach(ch => {
    ch.addEventListener('change', () => {
      ch.checked ? _selected.add(ch.dataset.email) : _selected.delete(ch.dataset.email);
      syncBulk();
      ch.closest('tr').classList.toggle('sel', ch.checked);
    });
  });
  syncBulk();
}

function syncBulk(){
  const n = _selected.size;
  $('#bulkBar').classList.toggle('on', n > 0);
  $('#bulkInfo').textContent = `Выбрано: ${n}`;
  const vis = visibleUsers();
  $('#selAll').checked = vis.length > 0 && vis.every(u => _selected.has(u.email));
}

$('#admSearch').addEventListener('input', renderUsers);
$('#admFilter').addEventListener('change', renderUsers);

$('#selAll').addEventListener('change', e => {
  // выделяем/снимаем только то, что сейчас видно (с учётом поиска и фильтра)
  visibleUsers().forEach(u => e.target.checked
    ? _selected.add(u.email) : _selected.delete(u.email));
  renderUsers();
});

$('#bulkExtend').onclick = async () => {
  const days = parseInt($('#bulkDays').value || '0', 10);
  if(!days){ alert('Укажи, сколько дней добавить (можно отрицательное — отнять).'); return; }
  const emails = [..._selected];
  const word = days > 0 ? `добавить ${days} дн.` : `отнять ${Math.abs(days)} дн.`;
  if(!confirm(`Выбрано аккаунтов: ${emails.length}.\n\n${word} каждому?`)) return;
  const j = await adminPost('/api/admin/extend', {emails, days});
  if(j.errors && j.errors.length) alert('Часть не удалась:\n' + j.errors.join('\n'));
  else alert(`Готово. Обновлено аккаунтов: ${j.count}.`);
  await loadUsers(); loadMe();
};

$('#bulkDelete').onclick = async () => {
  const emails = [..._selected];
  if(!confirm(`Удалить ${emails.length} аккаунт(ов)?\n\n`
      + emails.join('\n') + '\n\nОни потеряют доступ.')) return;
  const j = await adminPost('/api/admin/bulk-delete', {emails});
  if(j.errors && j.errors.length) alert('Часть не удалась:\n' + j.errors.join('\n'));
  _selected.clear();
  loadUsers();
};

async function extendUser(email, days){
  const j = await adminPost('/api/admin/extend', {email, days});
  if(!j.ok) alert(j.error); else { loadUsers(); loadMe(); }
}
async function resetPass(email){
  const p = prompt(`Новый пароль для ${email} (минимум 6 символов):`);
  if(!p) return;
  const j = await adminPost('/api/admin/password', {email, password:p});
  alert(j.ok ? `Пароль изменён.\n\nПочта: ${email}\nПароль: ${p}` : j.error);
}
async function delUser(email){
  if(!confirm(`Удалить пользователя ${email}? Он потеряет доступ.`)) return;
  const j = await adminPost('/api/admin/delete', {email});
  if(!j.ok) alert(j.error); else loadUsers();
}

loadMe();

// Если задача осталась в работе (страницу перезагрузили) — снова следим за ней.
fetch('/api/jobs').then(r=>r.json()).then(list=>{
  const live = (list||[]).find(j => j.status==='queued' || j.status==='running');
  if(live) watchJob(live.id);
});
loadResults();

// --- обработчики событий ---
// Встроенные onclick= в разметке запрещены политикой безопасности (CSP):
// без них браузер блокирует любой скрипт, внедрённый через содержимое
// страницы. Поэтому действия помечены data-атрибутами, а обрабатываются
// здесь — одним слушателем на весь документ, включая элементы, созданные
// уже после загрузки.
const ACTIONS = {
  stopJob:    (arg) => stopJob(arg),
  deleteClip: (arg) => deleteClip(arg),
  extendUser: (arg, el) => extendUser(arg, parseInt(el.dataset.days, 10)),
  resetPass:  (arg) => resetPass(arg),
  delUser:    (arg) => delUser(arg),
};

document.addEventListener('click', ev => {
  const el = ev.target.closest('[data-act]');
  if(!el) return;
  const fn = ACTIONS[el.dataset.act];
  if(!fn) return;
  ev.preventDefault();
  fn(el.dataset.arg, el);
});

document.addEventListener('change', ev => {
  if(ev.target.id === 'notifyCb') toggleNotify(ev.target);
});
