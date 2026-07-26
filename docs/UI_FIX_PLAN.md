# UI Fix Plan — Simple Agentic RAG Web UI

> แผนแก้ไข UI ตามผล UX/UI audit (อ่านจากโค้ดจริงใน `web/` ณ commit ปัจจุบัน)
> อ้างอิง issue id: UX-101 ถึง UX-107 · จัดลำดับตาม business impact ไม่ใช่ technical severity
>
> ไฟล์ที่เกี่ยวข้อง: `web/index.html` · `web/styles.css` · `web/tokens.css` · `web/app.js` · `web/api.js`
>
> หลักการคุมงานทั้งแผน (ห้ามละเมิด):
> - ทุกค่าที่มองเห็นต้อง resolve เป็น token จาก `tokens.css` — ห้าม hardcode hex/px ใน component CSS
> - `#live-region` เป็น aria-live region เดียวของหน้า — ทุก announcement ผ่าน `announce()` เท่านั้น
> - Motion ทุกตัววิ่งผ่าน `--duration-*` token เพื่อให้ reduced-motion ยุบที่ token layer

---

## ภาพรวมลำดับงาน

| Phase | Issue | เรื่อง | Severity | Effort (ประมาณ) |
|-------|-------|--------|----------|------------------|
| 1 | UX-101 | ปุ่ม "Switch to mock data" ใน error banner | High | ~1 ชม. |
| 1 | UX-102 | แก้ markup `<h2>` ซ้อนใน `<button>` (accordion pattern) | Medium | ~1–2 ชม. |
| 2 | UX-103 | Inline validation ตอน submit query ว่าง | Medium | ~1 ชม. |
| 2 | UX-104 | จำกัด line length ของ answer body | Low | ~15 นาที |
| 2 | UX-107 | Feedback เมื่อ clipboard ล้มเหลว | Low | ~30 นาที |
| 3 | UX-106 | Source toggle: role="switch" + คำอธิบาย score ที่มองเห็นได้ | Low | ~1 ชม. |
| 3 | UX-105 | Touch-target pass สำหรับ mobile | Low | ~1–2 ชม. |
| 4 | — | Verification + regression checklist | — | ~1 ชม. |

Phase 1 คือของที่กระทบ demo flow ตรงที่สุด ควรทำก่อนและทำจบเป็น commit แยกจาก polish

---

## Phase 1 — Recovery flow + Semantics (High impact)

### UX-101 · เพิ่มปุ่ม "Switch to mock data" ใน error banner

**ปัญหา** — NETWORK error message ([api.js:191-192](../web/api.js)) เขียนว่า
*"Start the Python service, or switch back to mock data."* แต่ `.error-actions`
([index.html:138-141](../web/index.html)) มีแค่ปุ่ม Retry — action ที่ข้อความสัญญาไว้ไม่มีให้กด
ผู้ใช้ต้องรู้เองว่า source pill มุมขวาบนกดสลับได้

**แนวทาง** — เพิ่มปุ่มที่สอง แสดงเฉพาะเมื่อเข้าเงื่อนไข แล้วสลับ + รันซ้ำในคลิกเดียว

**ขั้นตอน**

1. `web/index.html` — เพิ่มปุ่มใน `.error-actions` ก่อน `#retry-status`:

   ```html
   <div class="error-actions">
     <button type="button" id="retry-button" class="btn-secondary">Retry</button>
     <button type="button" id="switch-mock-button" class="btn-secondary" hidden>
       Switch to mock data
     </button>
     <span class="error-retry-status" id="retry-status"></span>
   </div>
   ```

2. `web/app.js` — ลงทะเบียน element ใน `el` map:

   ```js
   switchMockButton: document.getElementById("switch-mock-button"),
   ```

3. `web/app.js` — ใน `showError()` (หลังบรรทัด `el.errorState.hidden = false;`)
   เปิดปุ่มตามเงื่อนไข: error เป็น NETWORK **และ** ตอนนี้อยู่ live mode

   ```js
   el.switchMockButton.hidden = !(
     classifyError(error) === "NETWORK" && api.getMode() === "live"
   );
   ```

4. `web/app.js` — wiring (วางใกล้ `el.retryButton.addEventListener`):

   ```js
   el.switchMockButton.addEventListener("click", function () {
     api.setMode("mock");
     renderSourceToggle();
     announce("Data source set to mock data. Re-running the workflow.");
     if (state.query) {
       state.retryCount = 0; // mock ไม่ใช่ retry เดิม — reset backoff
       run(state.query);
     }
   });
   ```

5. อย่าลืมซ่อนปุ่มเมื่อเริ่ม run ใหม่ — ใน `run()` ตรงบล็อกที่ซ่อน error state:

   ```js
   el.switchMockButton.hidden = true;
   ```

**เหตุผลการออกแบบ**
- ใช้ `.btn-secondary` เดิม ไม่สร้าง style ใหม่ — Retry ยังเป็น action แรก (ซ้ายสุด) เพราะกรณี backend เพิ่งสตาร์ตเสร็จ Retry คือทางที่ถูก
- `state.retryCount = 0` เพราะการสลับ source เป็น run ใหม่ ไม่ควรโดน exponential backoff ของ retry เดิม
- `announce()` เพื่อให้ screen reader รู้ว่า mode เปลี่ยนและกำลังรันใหม่

**Acceptance criteria**
- [ ] อยู่ mock mode → error อื่น ๆ (เช่น RUNTIME) → ไม่เห็นปุ่ม
- [ ] อยู่ live mode → NETWORK error → เห็นปุ่ม กดแล้ว: pill header เปลี่ยนเป็น "Mock data", workflow รันซ้ำด้วย query เดิมทันที, error banner หาย
- [ ] กด Retry ตามปกติ → ปุ่ม switch ยังทำงานร่วมกับ backoff ได้ (switch ไม่ถูก disable ตาม backoff)

---

### UX-102 · แก้ markup `<h2>` ซ้อนใน `<button>` เป็น accordion pattern มาตรฐาน

**ปัญหา** — ทั้ง 5 step ([index.html:156-164](../web/index.html) และ step อื่น)
มี `<h2 class="step-title">` อยู่ **ภายใน** `<button class="step-toggle">` —
invalid HTML (button รับได้เฉพาะ phrasing content) และ button จะ flatten ลูกหลาน
ทำให้ heading หายจาก document outline → ผู้ใช้ screen reader navigate ด้วย
heading key ข้าม step ทั้งหมด

**แนวทาง** — ใช้ [ARIA Accordion Pattern](https://www.w3.org/WAI/ARIA/apg/patterns/accordion/):
heading ครอบ button ไม่ใช่ button ครอบ heading

**ขั้นตอน**

1. `web/index.html` — เปลี่ยนโครงสร้าง header ของ **ทั้ง 5 step** จาก:

   ```html
   <button type="button" class="step-toggle" aria-expanded="true" aria-controls="step-body-query">
     <svg class="chevron" ...>...</svg>
     <span class="step-heading">
       <span class="step-eyebrow">Step 1</span>
       <h2 class="step-title">User Query</h2>
     </span>
   </button>
   ```

   เป็น:

   ```html
   <h2 class="step-h">
     <button type="button" class="step-toggle" aria-expanded="true" aria-controls="step-body-query">
       <svg class="chevron" ...>...</svg>
       <span class="step-heading">
         <span class="step-eyebrow">Step 1</span>
         <span class="step-title">User Query</span>
       </span>
     </button>
   </h2>
   ```

   หมายเหตุ: `step-eyebrow` อยู่ใน button ต่อได้ (เป็น phrasing content) —
   accessible name ของ button จะเป็น "Step 1 User Query" ซึ่งโอเค

2. `web/styles.css` — เพิ่ม reset ให้ heading ตัวใหม่ (h2 ใน list `h1, h2, p, ...`
   ที่ margin:0 อยู่แล้ว แต่ต้องคุม font ไม่ให้ h2 default ทับ):

   ```css
   .step-h {
     display: contents; /* ให้ button จัด layout ใน .step-head เหมือนเดิม */
     font: inherit;     /* กัน UA h2 font-size/weight รั่วเข้า button */
   }
   ```

   > ถ้าเจอปัญหา `display: contents` กับ browser เก่า ให้ fallback เป็น
   > `.step-h { margin: 0; min-width: 0; display: flex; }` แทน — ตรวจ layout ด้วยตาทั้งสองแบบ

3. ตรวจว่า `.step-title` selector ยังทำงาน (เปลี่ยนจาก `h2.step-title` เป็น
   `span.step-title` — ใน [styles.css:772](../web/styles.css) ใช้ class selector อยู่แล้ว จึงไม่ต้องแก้)
   และ rule `.step[data-state="waiting"] .step-title` ([styles.css:711-714](../web/styles.css)) ยังจับถูกตัว

4. ตรวจ `web/app.js` — `stageEls` cache ใช้ `step.querySelector(".step-toggle")`
   และ delegation ใน `el.pipeline` ใช้ `.closest(".step-toggle")` / `.closest(".step-head")` —
   ทั้งคู่ไม่ผูกกับโครง h2 เดิม จึง **ไม่ต้องแก้ JS** แต่ต้อง regression-test การคลิก

**Acceptance criteria**
- [ ] HTML validate ผ่าน (ไม่มี heading ใน button)
- [ ] VoiceOver/NVDA: กด H วนได้ครบ h1 → h2 ทั้ง 5 step (+ h2 ของ empty/error state)
- [ ] คลิก header/chevron ยัง toggle collapse ได้เหมือนเดิม ทั้งคลิกที่ปุ่มตรง ๆ และคลิกพื้นที่ header
- [ ] Visual ไม่เปลี่ยน (เทียบ screenshot ก่อน/หลังทั้ง light/dark, มือถือ/desktop)

---

## Phase 2 — Feedback ที่มองเห็นได้ + Readability

### UX-103 · Inline validation ตอน submit query ว่าง

**ปัญหา** — submit handler ([app.js:910-915](../web/app.js)) กรณี query ว่าง:
มีแค่ `focus()` + `announce()` ซึ่งลง live region แบบ `.sr-only` —
ผู้ใช้ sighted ไม่เห็นอะไรเลย ดูเหมือนปุ่ม Run เสีย

**ขั้นตอน**

1. `web/index.html` — เพิ่ม element ข้อความใต้ `.query-row` (ใน `.query-card`):

   ```html
   <p class="field-error" id="query-error" hidden>Enter a question first.</p>
   ```

   และเพิ่ม `aria-describedby="query-error"` ให้ `#query-input`
   (คง attribute ไว้ตลอด — ตอน `hidden` browser จะไม่อ่านให้เอง ซึ่งคือพฤติกรรมที่ต้องการ)

2. `web/styles.css` — ใช้ warning token (นี่คือ validation ไม่ใช่ system failure —
   สงวน danger ไว้ให้ workflow error):

   ```css
   .field-error {
     font-size: var(--type-caption-size);
     line-height: var(--type-caption-line);
     font-weight: var(--type-caption-weight);
     color: var(--color-warning-fg);
   }
   ```

3. `web/app.js` — ใน submit handler:

   ```js
   var query = el.input.value.trim();
   if (!query) {
     el.queryError.hidden = false;
     el.input.focus();
     announce("Enter a question before running the workflow.");
     return;
   }
   el.queryError.hidden = true;
   ```

   และเพิ่ม listener เคลียร์ทันทีที่เริ่มพิมพ์:

   ```js
   el.input.addEventListener("input", function () {
     el.queryError.hidden = true;
   });
   ```

   (ลงทะเบียน `queryError: document.getElementById("query-error")` ใน `el` map)

**Acceptance criteria**
- [ ] กด Run ตอนว่าง → เห็นข้อความ warning ใต้ input, focus อยู่ที่ input
- [ ] พิมพ์ตัวแรก → ข้อความหาย
- [ ] Screen reader ได้ยินทั้งจาก live region และ `aria-describedby` ตอน focus input

---

### UX-104 · จำกัด line length ของ answer body

**ปัญหา** — `.answer-body` ([styles.css:1252-1260](../web/styles.css)) กว้างตามการ์ด
(~760px content ที่ font 14px ≈ 100+ ตัวอักษร/บรรทัด) เกินช่วงอ่านสบาย 50–75ch

**ขั้นตอน**

1. `web/tokens.css` — เพิ่ม token (ไปไว้ section 6 · COMPONENT SIZES):

   ```css
   --size-prose-max: 72ch; /* answer prose measure — meta bar stays full width */
   ```

2. `web/styles.css` — จำกัดเฉพาาะ prose ไม่ใช่ทั้ง body (meta bar / border ยังเต็มการ์ด):

   ```css
   .answer-body p,
   .answer-body ul {
     max-width: var(--size-prose-max);
   }
   ```

**Acceptance criteria**
- [ ] Desktop: บรรทัดคำตอบไม่เกิน ~72ch, meta bar (`#answer-meta`) ยังเต็มความกว้าง
- [ ] Mobile: ไม่มีผล (การ์ดแคบกว่า 72ch อยู่แล้ว)
- [ ] Citation chip ยัง wrap ตามข้อความปกติ

---

### UX-107 · Feedback เมื่อ clipboard ล้มเหลว

**ปัญหา** — `copyToClipboard()` ([app.js:888-902](../web/app.js)):
ถ้า `navigator.clipboard` ไม่มี (เช่นเปิดผ่าน `file://`) หรือถูก block —
เงียบสนิท ปุ่ม Copy ดูพัง

**ขั้นตอน**

1. `web/app.js` — แก้ `copyToClipboard` ให้มี failure branch ครบทั้งสองทาง:

   ```js
   function copyToClipboard(text, button) {
     var flash = function (label, message) {
       var original = button.textContent;
       button.textContent = label;
       if (message) announce(message);
       global.setTimeout(function () {
         button.textContent = original;
       }, COPY_FEEDBACK_MS);
     };

     if (global.navigator.clipboard && global.navigator.clipboard.writeText) {
       global.navigator.clipboard.writeText(text).then(
         function () { flash("Copied"); },
         function () { flash("Copy failed", "Copying is blocked in this context."); }
       );
     } else {
       flash("Copy failed", "Clipboard is not available in this context.");
     }
   }
   ```

   หมายเหตุ: ไม่เพิ่ม `document.execCommand("copy")` fallback — deprecated แล้ว
   และ demo รันบน `http://localhost` (secure context) เป็นหลัก ข้อความตรงไปตรงมาดีกว่า
   ทางแก้จริงของผู้ใช้คือปุ่ม select ข้อความเอง

**Acceptance criteria**
- [ ] บน localhost: Copy → "Copied" (พฤติกรรมเดิมไม่เปลี่ยน)
- [ ] เปิดผ่าน `file://`: Copy → "Copy failed" + live region แจ้งเหตุผล แล้ว label เด้งกลับ
- [ ] ใช้ได้กับทั้ง 3 จุดที่เรียก: copy answer, copy snippet, copy trace id

---

## Phase 3 — Affordance + Touch polish

### UX-106 · Source toggle semantics + คำอธิบาย score ที่มองเห็นได้

**ปัญหา** — สองจุดพึ่ง `title` attribute (hover-only ไม่ถึง touch/keyboard):
1. `#source-toggle` ([index.html:45-53](../web/index.html)) — ไม่มี role/state สื่อว่าเป็น toggle; label บอกแค่ state ปัจจุบัน
2. `.snippet-score` ([app.js:517](../web/app.js)) — คำอธิบาย "Retrieval score (title-weighted term match)" อยู่ใน tooltip รายชิ้น

**ขั้นตอน — source toggle**

1. `web/index.html` — เพิ่ม semantics + prefix ที่มองเห็น:

   ```html
   <button type="button" id="source-toggle" class="source-pill"
           role="switch" aria-checked="false"
           title="Switch between bundled mock data and a live backend">
     <span class="source-pill-prefix">Source:</span>
     <span class="source-dot" aria-hidden="true"></span>
     <span id="source-label">Mock data</span>
   </button>
   ```

   Convention: `aria-checked="true"` = live backend (ฝั่งที่ "เปิดของจริง")

2. `web/styles.css`:

   ```css
   .source-pill-prefix {
     color: var(--color-text-tertiary);
     font-size: var(--type-caption-size);
   }
   ```

3. `web/app.js` — ใน `renderSourceToggle()` เพิ่ม:

   ```js
   el.sourceToggle.setAttribute("aria-checked", String(mode === "live"));
   ```

**ขั้นตอน — snippet score**

4. `web/app.js` — อธิบายครั้งเดียวที่ระดับ step แทน tooltip รายชิ้น:
   แก้ `.step-note` ของ Step 3 ใน `index.html` เป็น

   ```html
   <p class="step-note">
     Raw sections exactly as returned by the tool. Scores are title-weighted
     term-match values — higher is more relevant.
   </p>
   ```

   แล้ว **ลบ** `title="..."` ออกจาก `.snippet-score` ใน `renderEvidence()`
   (คงตัวเลข `score 3.2` ไว้เหมือนเดิม)

**Acceptance criteria**
- [ ] VoiceOver อ่าน source toggle เป็น "Source: Mock data, switch, off/on"
- [ ] สลับ mode แล้ว `aria-checked` ตามทัน (ทั้งคลิกที่ pill และผ่านปุ่ม UX-101)
- [ ] คำอธิบาย score อ่านได้โดยไม่ต้อง hover ทั้ง desktop/mobile

---

### UX-105 · Touch-target pass สำหรับ mobile (< 768px)

**ปัญหา** — ปุ่มรองสูงจริง ~26–32px: `.btn-icon` (Copy), `.chip`, `.source-pill`
ผ่าน WCAG 2.5.8 ขั้นต่ำ (24px) แต่ต่ำกว่า 44px comfort บน touch
โดยหน้ามี mobile layout จริง (breakpoint 768px + compact bar)

**แนวทาง** — ขยาย **hit area** โดยไม่เปลี่ยน visual ด้วย pseudo-element
(วิธีขยาย padding จะทำ layout ขยับและชน design เดิม)

**ขั้นตอน**

1. `web/styles.css` — เพิ่มบล็อกใน media query mobile (สร้างใหม่ `@media (max-width: 767.98px)`
   วางไว้ก่อน section Responsive เดิม):

   ```css
   @media (max-width: 767.98px) {
     .btn-icon,
     .chip,
     .source-pill,
     .clamp-toggle {
       position: relative;
     }
     /* ขยาย hit area เป็น >=44px แบบไม่แตะ visual — จุดกดทับกันเองไม่ได้
        เพราะ gap ระหว่างปุ่มใน .chips/.snippet-head คือ 8px ต่อฝั่ง (4px+4px)
        พอดีกับส่วนขยาย 8px แนวตั้ง/แนวนอนต่อฝั่งของแต่ละปุ่ม */
     .btn-icon::after,
     .chip::after,
     .source-pill::after,
     .clamp-toggle::after {
       content: "";
       position: absolute;
       inset: calc(-1 * var(--space-2)) calc(-1 * var(--space-1));
     }
   }
   ```

   > ข้อควรระวัง: `.snippet-head` มีปุ่ม Copy ชิดกับ tag `raw` — tag เป็น `<span>`
   > ไม่ใช่ interactive จึงไม่มีปัญหา overlap แต่ให้ตรวจ `.chips` ที่ chip เรียงติดกัน
   > (gap 8px) ว่า hit area ขยายแล้วไม่ทับกัน: inset แนวนอน -4px ต่อฝั่ง → ช่องว่างเหลือ 0px พอดี
   > ถ้าทับให้ลดเหลือ `calc(-1 * var(--space-1))` ทั้งสองแกน

2. ตรวจของที่ **ไม่ต้องแก้**: `.step-toggle` กดได้ทั้ง header (ใหญ่พอ),
   `.btn-primary`/`.btn-secondary` สูง ~44/38px, `.cite` เป็น inline target
   ใน paragraph — ได้รับ exemption จาก WCAG 2.5.8

**Acceptance criteria**
- [ ] iOS Safari + Android Chrome (หรือ DevTools device mode): กด Copy/chip/source pill ที่ขอบ ๆ ติดง่ายขึ้น ไม่มีการกดผิดปุ่มข้างเคียง
- [ ] Visual ทุก breakpoint เหมือนเดิม 100%
- [ ] Desktop (≥768px) ไม่ได้รับผล

---

## Phase 4 — Verification & Regression

### เครื่องมือ

```bash
# เปิด demo (โปรเจกต์นี้ serve ผ่าน Python backend หรือเปิด static)
python main.py --serve  # หรือวิธี serve ที่ใช้อยู่เดิม
```

- **Lighthouse (Chrome DevTools)** — Accessibility category: เป้า ≥ 95 ทั้งสองหน้าหลัก (idle + result)
- **axe DevTools extension** — 0 critical/serious issues
- **HTML validation** — `https://validator.w3.org/#validate_by_input` วาง `index.html` (ตรวจ UX-102)

### Manual checklist (รันทุกข้อหลังจบแต่ละ Phase)

**Keyboard**
- [ ] Tab ไล่ครบ: input → Run → chips → source toggle → step toggles → Copy/cite/expand
- [ ] `Cmd/Ctrl+Enter` submit ได้จากทุกจุด focus
- [ ] Focus ring มองเห็นชัดทุก interactive element (token `--color-focus-ring`)

**Screen reader (VoiceOver อย่างน้อย)**
- [ ] Heading navigation (กด H): h1 → h2 ครบ 5 step
- [ ] Live region ประกาศ: running → completed/failed/no-evidence, mode switch, copy failed
- [ ] Source toggle อ่านเป็น switch พร้อม state

**States ทั้ง 5 ของ demo (mock mode)**
- [ ] Query ปกติ ("international travel") → 3 snippets + citations กดกระโดดได้
- [ ] "What is the refund policy?" → no-evidence path: banner + Rewrite/Show topics ทำงาน
- [ ] Live mode โดยไม่มี backend → NETWORK error → **เห็นปุ่ม Switch to mock data → กดแล้วรันจบ** (UX-101)
- [ ] Cancel กลางคัน → กลับ idle สะอาด
- [ ] Submit ค่าว่าง → เห็น inline warning (UX-103)

**Visual**
- [ ] Light/dark × desktop/mobile (375px) — เทียบกับ `screenshots/` เดิม จุดที่ต่างต้องมีเหตุผลจาก plan นี้เท่านั้น
- [ ] `prefers-reduced-motion` → ไม่มี animation, spinner เป็น dot นิ่ง

### ลำดับ commit ที่แนะนำ

```
fix(web): add switch-to-mock recovery action on network errors   (UX-101)
fix(web): valid accordion markup — heading wraps toggle button    (UX-102)
feat(web): visible empty-query validation message                 (UX-103)
style(web): cap answer prose measure at 72ch                      (UX-104)
fix(web): surface clipboard failures on copy buttons              (UX-107)
a11y(web): source toggle switch semantics + visible score note    (UX-106)
a11y(web): expand touch hit areas on mobile                       (UX-105)
```

---

## สิ่งที่ตัดสินใจ "ไม่ทำ" ในรอบนี้ (พร้อมเหตุผล)

| เรื่อง | เหตุผล |
|--------|--------|
| Segmented control (Mock \| Live) เต็มรูปแบบ | role="switch" + prefix "Source:" (UX-106) ปิด gap หลักแล้ว — segmented control เปลี่ยน layout header บน mobile และได้ผลเพิ่มน้อย |
| `light-dark()` fallback สำหรับ browser ก่อน mid-2024 | เป็น demo ที่คุมสภาพแวดล้อมได้ — document ไว้ใน DESIGN_NOTES แล้ว ถ้าจะรองรับจริงต้อง expand เป็น theme block ซ้ำทั้งไฟล์ (งานใหญ่ ผลตอบแทนต่ำ) |
| `execCommand` clipboard fallback | Deprecated — แจ้ง "Copy failed" ตรงไปตรงมาดีกว่าพึ่ง API ที่กำลังถูกถอด |
| CI a11y check (axe ใน pipeline) | ควรทำ แต่เป็นงาน infra แยกจาก UI fix — เสนอเป็น ticket ถัดไป |
