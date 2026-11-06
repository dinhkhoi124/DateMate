## Thay đổi gì, vì sao

<!-- 1–3 câu. Link issue nếu có: Closes #.. -->

## Loại thay đổi

- [ ] Ingest / đọc dữ liệu
- [ ] Parser / intent / gate
- [ ] Executor / luật rulebook
- [ ] LLM / prompt
- [ ] Eval / tooling / Docker / docs

## Kết quả đo (bắt buộc nếu đổi `src/`)

| | Trước | Sau |
|---|---|---|
| dev PA (đúng / sai / abstain) | | |
| proxy shop lạ (`eval/synth`, nếu đổi ingest) | | |
| holdout PA (chỉ khi chốt milestone) | | |

## Checklist

- [ ] `pytest tests -q` xanh, `ruff check` sạch (CI sẽ kiểm lại)
- [ ] **Số câu sai trên dev không tăng.** Nếu PA giảm, đã giải thích ở trên
- [ ] Đã ghi một dòng vào `docs/EXPERIMENTS.md`
- [ ] Luật rulebook mới có docstring trích mục rulebook và có test
- [ ] Quyết định kiến trúc / model / dependency mới đã ghi vào `docs/DECISIONS.md`
- [ ] **Tôi không đọc câu hỏi public và không mở dữ liệu / câu hỏi của shop holdout** khi làm PR này
- [ ] Không hardcode theo `id`, `business_id` hay tên shop
- [ ] Đã có ít nhất 1 người khác xem PR (comment hoặc approve) trước khi merge
