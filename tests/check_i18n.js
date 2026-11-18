// Kiểm tra: 3 ngôn ngữ có cùng tập key, và mọi data-i18n trong index.html đều có bản dịch.
const fs = require('fs');
const path = process.argv[2];

const src = fs.readFileSync(path + '/static/i18n.js', 'utf8');
const html = fs.readFileSync(path + '/templates/index.html', 'utf8');
const app = fs.readFileSync(path + '/static/app.js', 'utf8');

// Chạy i18n.js trong môi trường giả lập tối thiểu
const sandbox = {
    document: {
        addEventListener() {},
        querySelectorAll() { return []; },
        documentElement: {},
        dispatchEvent() {},
    },
    localStorage: { getItem() { return null; }, setItem() {} },
    CustomEvent: function () {},
};
sandbox.window = sandbox;
const vm = require('vm');
vm.createContext(sandbox);
vm.runInContext(src, sandbox);

const I18N = vm.runInContext('I18N', sandbox);
const langs = Object.keys(I18N);
const keysets = langs.map((l) => new Set(Object.keys(I18N[l])));
let bad = 0;

langs.forEach((l, i) => {
    langs.forEach((m, j) => {
        if (i >= j) return;
        for (const k of keysets[i]) {
            if (!keysets[j].has(k)) { console.log(`THIEU key "${k}" trong ${m} (co trong ${l})`); bad++; }
        }
        for (const k of keysets[j]) {
            if (!keysets[i].has(k)) { console.log(`THIEU key "${k}" trong ${l} (co trong ${m})`); bad++; }
        }
    });
});

// Mọi key dùng trong HTML / JS phải tồn tại
const used = new Set();
for (const m of html.matchAll(/data-i18n(?:-placeholder|-alt)?="([^"]+)"/g)) used.add(m[1]);
for (const m of app.matchAll(/\bt\(\s*'([^']+)'/g)) used.add(m[1]);

for (const k of used) {
    if (!keysets[0].has(k)) { console.log(`KHONG CO ban dich cho key "${k}"`); bad++; }
}

// id do app.js tự tạo lúc chạy (không cần có sẵn trong HTML)
const DYNAMIC_IDS = new Set(['toast-container']);
// Mọi id được app.js gọi qua getElementById phải có trong HTML
const ids = new Set();
for (const m of html.matchAll(/id="([^"]+)"/g)) ids.add(m[1]);
for (const m of app.matchAll(/getElementById\(\s*'([^']+)'/g)) {
    if (!ids.has(m[1]) && !DYNAMIC_IDS.has(m[1])) { console.log(`KHONG CO id="${m[1]}" trong index.html`); bad++; }
}

console.log(bad === 0 ? `OK — ${langs.join('/')}, ${keysets[0].size} key, ${used.size} key dang dung` : `${bad} van de`);
