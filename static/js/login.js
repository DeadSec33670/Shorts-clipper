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
const err = document.getElementById('err');
if(new URLSearchParams(location.search).get('expired')){
  err.textContent = 'Срок доступа истёк. Обратитесь к администратору.';
  err.style.display = 'block';
}
document.getElementById('f').addEventListener('submit', async e => {
  e.preventDefault();
  const btn = document.getElementById('btn');
  btn.disabled = true; err.style.display='none';
  const r = await fetch('/api/login', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({
      email: document.getElementById('email').value,
      password: document.getElementById('password').value })
  });
  const j = await r.json();
  if(j.ok){ location.href = '/'; }
  else { err.textContent = j.error || 'Ошибка входа'; err.style.display='block'; btn.disabled=false; }
});