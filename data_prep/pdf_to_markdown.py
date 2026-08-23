# pdf_to_markdown.py
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# 示例：安装 requests
install_package("PyMuPDF")

import torch
import fitz  # PyMuPDF
from PIL import Image, ImageOps
import os
import yaml
import re
from collections import Counter
import subprocess, os
from bs4 import BeautifulSoup
import logging
import unicodedata
# 动态将当前脚本的上一级目录（即项目根目录 /workspace）加入 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from factory.model_factory import ModelFactory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

def print_mem(tag=""):
    # 显存
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserv = torch.cuda.memory_reserved() / 1024**2
        logging.info(f"[{tag}] 显存 allocated={alloc:.0f}MB reserved={reserv:.0f}MB")
    # RAM
    try:
        result = subprocess.run(['free', '-m'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if 'Mem:' in line:
                parts = line.split()
                logging.info(f"[{tag}] RAM total={parts[1]}MB used={parts[2]}MB free={parts[3]}MB available={parts[6]}MB")
    except (subprocess.SubprocessError, OSError, IndexError) as e:
        logging.debug(f"[{tag}] 获取 RAM 信息失败（非致命）: {e}")


class MarkdownProcessor:
    def __init__(self, prompt_hub_path="prompt_hub.yaml", cache_dir="/workspace/hf-conda/hf_cache/hub"):
        self.prompt_hub_path = prompt_hub_path
        # 实例化 ModelFactory（自动建立软链接并加载 prompt 资产）
        self.factory = ModelFactory(prompt_hub_path=prompt_hub_path, cache_dir=cache_dir)
        self.prompts = self.factory.prompts

    def load_prompts(self):
        """复用 ModelFactory 的 prompt 资源"""
        return self.factory.prompts

    def resolve_model_path(self, short_name, cache_dir="/workspace/hf-conda/hf_cache/hub"):
        """委托给 ModelFactory 统一进行路径解析"""
        return self.factory.resolve_model_path(short_name)

    def get_vlm_model(self, vlm='Qwen--Qwen3-VL-32B-Instruct'):
        """委托 ModelFactory 统一进行多模态模型加载与预热"""
        return self.factory.get_vlm_model(vlm_short_name=vlm)

    # def _destroy_model(self):
    #     """委托 ModelFactory 主动熔断销毁 VLM 显存资源"""
    #     self.factory.destroy_vlm_model()

    def _destroy_model(self):
        # 正确：在包含 factory 的类中进行调用
        if hasattr(self, 'factory') and self.factory:
            self.factory.destroy_vlm_model()
            self.model = None

    @staticmethod
    def process_html_to_flat_html(
            html_content,
            header_rows=None,
            ffill_columns=None,
            header_sep="_",
            include_nested_tables=False,
            max_cols_safety=1024,
            cell_text_fn=None,
            is_header_row_fn=None,
        ):
            """
            将带有 rowspan/colspan 合并单元格的 HTML 表格扁平化为简单的平铺表格。

            参数
            ----------
            html_content : str
                原始的 HTML 表格内容。
            header_rows : int | None
                强制指定前几行为表头行。最高优先级 —— 设置后，is_header_row_fn 和内置的启发式算法都将被跳过。
            ffill_columns : None | "all" | list[int] | callable(col_idx) -> bool
                控制对 *空白* 物理单元格的向后填充（forward-fill）。这些单元格不是 HTML 显示指定的 rowspan/colspan
                （即“隐式”合并，在 VLM 提取的表格中很常见，其中看起来合并的单元格并没有被标记 rowspan 属性）。
                None/"none" = 不填充（默认，最安全）。
                "all" = 填充所有列。
                list[int] = 仅填充这些列索引。
                可调用对象（函数）允许你根据自己的逻辑决定每列是否填充（例如根据表头文本）。
            header_sep : str
                拼接多行表头时使用的分隔符。
            include_nested_tables : bool
                如果为 False（默认），则在提取文本之前，将单元格内嵌套的 <table> 内容剥离，以避免混杂拼接。
            max_cols_safety : int
                表格宽度的防御性限制，以避免在病态 HTML 上陷入死循环。
            cell_text_fn : callable(cell, tag_name, default_text) -> str | None
                用于自定义每个单元格文本提取/清洗的闭包（例如剥离管道占位符如 "§"，或规范化数字）。
                返回 None 以保留该单元格的默认文本。
            is_header_row_fn : callable(row_idx, row_texts, has_th, num_rows) -> bool | None
                用于自定义表头行检测的闭包。每行调用一次（仅在未设置 header_rows 时）。
                返回 True/False 以强制决定该行是否为表头，或返回 None 以回退到内置的启发式算法
                （包含 <th> 标签，或紧随已确认表头行之后的非数值文本）。
                row_texts 是该行中每一列按顺序扁平化后的文本。
            """

            # ========================================================
            # Inner Helper Functions (To keep the global namespace clean)
            # ========================================================
            def _safe_span(value, default=1, min_val=1, cap=1000):
                """
                鲁棒地解析 rowspan/colspan。
                VLM 或爬虫生成的 HTML 可能包含非数字、空值或零值 —— 绝对不能让它导致流水线崩溃。
                """
                if value is None:
                    return default
                m = re.search(r"\d+", str(value))
                if not m:
                    return default
                n = int(m.group())
                if n < min_val:
                    return default
                return min(n, cap)

            def _looks_numeric(text):
                """判断文本是否看起来像是一个数值"""
                if not text:
                    return False
                t = text.strip().replace(",", "").replace("%", "")
                if t == "":
                    return False
                try:
                    float(t)
                    return True
                except ValueError:
                    return False

            def _cell_text(cell, include_nested, text_fn=None):
                """
                提取单元格文本。
                默认情况下，会剥离任何嵌套的 <table>，以避免其子单元格内容被拼接到父单元格的文本中。
                """
                if not include_nested:
                    for inner in cell.find_all("table"):
                        inner.extract()
                default_text = cell.get_text(separator=" ", strip=True)
                if text_fn is not None:
                    custom = text_fn(cell, cell.name, default_text)
                    if custom is not None:
                        return custom
                return default_text

            # ========================================================
            # 核心处理逻辑
            # ========================================================
            soup = BeautifulSoup(html_content, "html.parser")
            table = soup.find("table")
            if not table:
                return ""

            # find_all('tr') 也会递归搜索嵌套表；这里仅保留其最近一级父 <table> 是当前表格的行。
            rows = [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]
            num_rows = len(rows)
            if num_rows == 0:
                return "<table border='1'>\n</table>"

            # ========================================================
            # Step 1: 以 (行, 列) 为键构建网格字典。
            # 这样避免了必须提前预测总列数（当先前的 rowspan 占用了后续更宽行所需的列时，
            # 旧的固定宽度二维数组会少算实际的表格宽度）。
            # ========================================================
            grid = {}
            is_virtual = set()
            max_c = -1

            for r_idx, row in enumerate(rows):
                c_idx = 0
                # recursive=False: 单元格内嵌套的 <table>，其自身的 <td>/<th> 绝对不能算作当前主表的列。
                cells = row.find_all(["td", "th"], recursive=False)

                for cell in cells:
                    guard = 0
                    while (r_idx, c_idx) in grid:
                        c_idx += 1
                        guard += 1
                        if guard > max_cols_safety:
                            break

                    rowspan = _safe_span(cell.get("rowspan", 1), cap=num_rows)
                    colspan = _safe_span(cell.get("colspan", 1), cap=max_cols_safety)
                    cell_text = _cell_text(cell, include_nested_tables, cell_text_fn)
                    tag_name = cell.name

                    for r_offset in range(rowspan):
                        target_r = r_idx + r_offset
                        if target_r >= num_rows:
                            break
                        for c_offset in range(colspan):
                            target_c = c_idx + c_offset
                            if target_c >= max_cols_safety:
                                break
                            grid[(target_r, target_c)] = (tag_name, cell_text)
                            if r_offset > 0 or c_offset > 0:
                                is_virtual.add((target_r, target_c))
                            if target_c > max_c:
                                max_c = target_c

                    c_idx += colspan

            num_cols = max_c + 1 if max_c >= 0 else 0

            def grid_get(r, c):
                return grid.get((r, c))
            
            # ========================================================
            # Step 2: 表头行检测（已根据用户需求优化）
            # 策略：从第一行开始判断，直到遇到“没有空单元格”的行，
            # 将该行作为表头的最后一行（往上包含该行都是表头），随后中断检测。
            # ========================================================
            if header_rows is not None:
                n = max(0, header_rows)
                if num_rows > 1:
                    n = min(n, num_rows - 1)  # 总是至少留一行作为数据行
                else:
                    n = min(n, num_rows)
                header_row_indices = list(range(n))
            else:
                header_row_indices = []
                for r_idx in range(num_rows):
                    row_cells = [grid_get(r_idx, c) for c in range(num_cols)]
                    row_texts = [c[1] if c else "" for c in row_cells]
                    has_th = any(c and c[0] == "th" for c in row_cells if c)

                    decision = None
                    if is_header_row_fn is not None:
                        decision = is_header_row_fn(r_idx, row_texts, has_th, num_rows)

                    if decision is None:
                        # 检查当前行是否存在空单元格（隐式或显式合并导致的空白）
                        has_empty_cell = any(not t.strip() for t in row_texts)
                        
                        # 安全防御：如果表格有多行，且当前已经是最后一行，强制判定为非表头（留作数据行）
                        if num_rows > 1 and r_idx == num_rows - 1:
                            decision = False
                        else:
                            decision = True

                        if decision:
                            header_row_indices.append(r_idx)
                        
                        # 【核心改动】如果这一行没有空单元格，说明它是干净的完整行
                        # 将其作为表头的最后一行，判定结束，直接跳出循环
                        if not has_empty_cell:
                            break
                    else:
                        if decision:
                            header_row_indices.append(r_idx)
                        else:
                            break
            data_start_idx = (max(header_row_indices) + 1) if header_row_indices else 0

            # ========================================================
            # Step 3: 自动检测“包含合并单元格”的列
            # 逻辑：只要这一列中有任何一个单元格属于 is_virtual (来自真实的 rowspan/colspan)
            # 或者该列的原始 tag 带有 rowspan，就将这一列标记为“需强制填充列”
            # ========================================================
            auto_ffill_cols = set()
            for (r, c) in is_virtual:
                auto_ffill_cols.add(c) # 记录下所有存在合并单元格的列索引

            def col_selected(c_idx):
                # 1. 如果用户显式传了 ffill_columns，优先听用户的
                if ffill_columns is not None and ffill_columns != "none":
                    if ffill_columns == "all":
                        return True
                    if callable(ffill_columns):
                        return bool(ffill_columns(c_idx))
                    if c_idx in ffill_columns:
                        return True
                
                # 2. 🌟 你的核心需求：如果该列识别到了合并单元格，该列自动开启全部向下填充
                if c_idx in auto_ffill_cols:
                    return True

                return False

            # ========================================================
            # Step 4: 填充（结合了“列级合并判定”与“粘性继承”）
            # ========================================================
            for c_idx in range(num_cols):
                last_tag, last_text = "td", ""
                for r_idx in range(data_start_idx, num_rows):
                    cell_data = grid_get(r_idx, c_idx)

                    # 情况 A：来自真实 HTML 属性 rowspan 的虚拟单元格 -> 必须继承
                    if (r_idx, c_idx) in is_virtual:
                        if last_text:
                            grid[(r_idx, c_idx)] = (last_tag, last_text)
                        continue

                    tag, text = cell_data if cell_data else ("td", "")

                    # 情况 B：普通空白 <td>
                    if text == "":
                        # 如果该列触发了“合并列判定”，或者被配置为需要填充
                        if col_selected(c_idx) and last_text:
                            grid[(r_idx, c_idx)] = (last_tag, last_text)
                        else:
                            last_tag, last_text = "td", ""
                    else:
                        # 情况 C：遇到了新的非空文本，更新上下文信息
                        last_tag, last_text = tag, text

            # # ========================================================
            # # Step 3: 填充。
            # #   - 显式合并（来自真实 rowspan/colspan 的虚拟合并单元格）：
            # #     总是继承源单元格的内容（防御性重断言）。
            # #   - 隐式合并（空白物理 <td>，无 rowspan 属性）：
            # #     仅对通过 ffill_columns 选中的列进行填充。
            # # ========================================================

            # def col_selected(c_idx):
            #     if ffill_columns is None or ffill_columns == "none":
            #         return False
            #     if ffill_columns == "all":
            #         return True
            #     if callable(ffill_columns):
            #         return bool(ffill_columns(c_idx))
            #     return c_idx in ffill_columns

            # for c_idx in range(num_cols):
            #     last_tag, last_text = "td", ""
            #     for r_idx in range(data_start_idx, num_rows):
            #         cell_data = grid_get(r_idx, c_idx)
            #         if (r_idx, c_idx) in is_virtual:
            #             if last_text:
            #                 grid[(r_idx, c_idx)] = (last_tag, last_text)
            #             continue

            #         tag, text = cell_data if cell_data else ("td", "")
            #         if text == "":
            #             if col_selected(c_idx) and last_text:
            #                 grid[(r_idx, c_idx)] = (last_tag, last_text)
            #                 # keep last_tag/last_text — same run continues
            #             else:
            #                 last_tag, last_text = "td", ""
            #         else:
            #             last_tag, last_text = tag, text

            # ========================================================
            # Step 5: 构建表头（将每列的多行表头进行拼接）
            # ========================================================
            final_headers = []
            for c_idx in range(num_cols):
                col_cells = []
                for r_idx in range(data_start_idx):
                    cell_data = grid_get(r_idx, c_idx)
                    col_cells.append(cell_data[1] if cell_data else "")

                cleaned = []
                for cell in col_cells:
                    if cell and (not cleaned or cell != cleaned[-1]):
                        cleaned.append(cell)

                final_headers.append(header_sep.join(cleaned) if cleaned else f"Column_{c_idx}")

            # ========================================================
            # Step 6: 渲染成扁平的 HTML
            # ========================================================
            html_lines = ["<table border='1'>", "  <tr>"]
            for header in final_headers:
                html_lines.append(f"    <th>{header}</th>")
            html_lines.append("  </tr>")

            for r_idx in range(data_start_idx, num_rows):
                html_lines.append("  <tr>")
                for c_idx in range(num_cols):
                    cell_data = grid_get(r_idx, c_idx)
                    html_lines.append(f"    <td>{cell_data[1] if cell_data else ''}</td>")
                html_lines.append("  </tr>")
            html_lines.append("</table>")

            return "\n".join(html_lines)
#--------------------------------------------------------------------------------------------

    def erase_template_noise(self, doc: fitz.Document, threshold=0.7):
        """
        1. 自动探测 PDF 页眉页脚模板，擦除重复噪声，输出干净的 PDF。
        """
        def _inner_normalize(t):
            if not t: return ""
            t = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff]', '', t).lower()
            return re.sub(r'\d+', '#', t)
        
        logging.info(f"开始处理 PDF: {doc.name}，输出干净 PDF...")
        
        if len(doc) == 0:
            doc.close()
            return None

        pw, ph = doc[0].rect.width, doc[0].rect.height
        total_pages = len(doc)

        # 动态探测全局水位线
        h_limit, f_limit = 0, ph
        if total_pages >= 2:
            first_page = doc[0]
            blocks = first_page.get_text("dict", sort=True)["blocks"]
            y_coords = sorted(list(set([b["bbox"][1] for b in blocks if "lines" in b])))
            check_range = min(10, total_pages)

            for y0 in y_coords:
                if y0 > ph * 0.25: break 
                curr_b = [b for b in blocks if b["bbox"][1] == y0][0]
                y1 = curr_b["bbox"][3]
                rect = fitz.Rect(0, 0, pw, y1 + 1)
                fp = _inner_normalize(first_page.get_text("text", clip=rect))
                matches = sum(1 for i in range(1, check_range) 
                            if _inner_normalize(doc[i].get_text("text", clip=rect)) == fp)
                if matches / (check_range - 1) >= threshold: h_limit = y1
                else: break

            for y0 in reversed(y_coords):
                if y0 < ph * 0.75: break 
                rect = fitz.Rect(0, y0 - 1, pw, ph)
                fp = _inner_normalize(first_page.get_text("text", clip=rect))
                matches = sum(1 for i in range(1, check_range) 
                            if _inner_normalize(doc[i].get_text("text", clip=rect)) == fp)
                if matches / (check_range - 1) >= threshold: f_limit = y0
                else: break
        else:
            h_limit, f_limit = 0, ph

        # logging.info(f"DEBUG - 动态边界: 页眉 {h_limit:.1f} | 页脚 {f_limit:.1f}")

        noise_re = re.compile(r"第\s*\d+\s*页|Page\s*\d+|\d{4}/\d{1,2}/\d{1,2}", re.I)
        fragments = []
        total_h = 0.0

        header_zone = fitz.Rect(0, 0, pw, h_limit + 1) if h_limit > 0 else None
        footer_zone = fitz.Rect(0, f_limit - 1, pw, ph) if f_limit < ph else None

        for page in doc:
            if header_zone: page.add_redact_annot(header_zone, fill=(1, 1, 1))
            if footer_zone: page.add_redact_annot(footer_zone, fill=(1, 1, 1))

            for b in page.get_text("blocks"):
                if b[6] == 0 and noise_re.search(b[4]):
                    r = fitz.Rect(b[:4])
                    if r.y1 < h_limit + 20 or r.y0 > f_limit - 20:
                        page.add_redact_annot(r, fill=(1, 1, 1))
            
            page.apply_redactions(images=0, graphics=1)
            page.clean_contents()
            body_rects = []
            for b in page.get_text("blocks"):
                if b[6] == 0 and b[4].strip():
                    r = fitz.Rect(b[:4])
                    if h_limit < r.y0 and r.y1 < f_limit: body_rects.append(r)
            for img in page.get_image_info():
                r = fitz.Rect(img["bbox"])
                if h_limit < r.y0 and r.y1 < f_limit: body_rects.append(r)

            if body_rects:
                union = body_rects[0]
                for r in body_rects[1:]: union |= r
                crop = fitz.Rect(0, max(h_limit, union.y0 - 0), pw, min(f_limit, union.y1 + 0))
                fragments.append((page.number, crop))
                total_h += crop.height

        if not fragments:
            doc.close()
            return None

        stitch_doc = None
        try:
            stitch_doc = fitz.open()
            long_page = stitch_doc.new_page(width=pw, height=total_h)
            
            curr_y = 0.0
            for p_no, c_rect in fragments:
                target_rect = fitz.Rect(0, curr_y, pw, curr_y + c_rect.height)
                long_page.show_pdf_page(target_rect, doc, p_no, clip=c_rect)
                curr_y += c_rect.height
                
            for p in stitch_doc:
                p.set_cropbox(p.rect)
                p.set_mediabox(p.rect)

            # stitch_doc.save(str('/workspace/hf-conda/RAG/问答机器人/finebi/函数专题/1_函数新手入门/aaaa.pdf'), garbage=4, deflate=True, clean=True)
            # 🛠️ 关键修复：把 stitch_doc 送进去，函数内部运行完会自动安全关闭它
            # logging.info(f"✅ 长 PDF 已缝合...")
            return stitch_doc
        except Exception as e:
            logging.info(f"⚠️ 处理 PDF 时发生错误: {e}")
            if stitch_doc:
                stitch_doc.close()
            return None
   
    def predict_pdf_native_single(self, clean_doc: fitz.Document, user_prompt: str, vlm='Qwen--Qwen3-VL-32B-Instruct') -> str:
        """
        2. 使用模型对单页 PDF 进行原生解析，返回 Markdown 文本, 方式为单页直接输入。
        """
        logging.info(f"📄 正在解析 PDF...")
        model, processor = self.get_vlm_model(vlm)
        logging.info("🔍 正在处理 PDF 页面并构建对话输入...")
        # 1. 加载 PDF 页面
        doc = clean_doc

        page = doc[0]
        
        # 保持 2 倍缩放确保文字清晰度 [cite: 7, 10, 42]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # 2. 图像预处理
        # A. 灰度化：消除背景色/水印干扰，强化表格框架
        image = image.convert("L")
        
        # B. 对比度拉伸：让白底更白，黑线更黑。cutoff=0.5 能有效滤除扫描噪点
        image = ImageOps.autocontrast(image, cutoff=0.5)
        
        # C. 动态长宽比调整 (Padding)
        w, h = image.size
        aspect_ratio = h / w
        
        # 如果比例超过 1.5 (偏长)，添加侧边白色填充，保持表格在画面中心的比例稳定
        if aspect_ratio > 1.5:
            target_width = int(h / 1.5)
            new_image = Image.new("L", (target_width, h), 255) # 纯白背景
            offset = (target_width - w) // 2
            new_image.paste(image, (offset, 0))
            image = new_image
        
        # D. 转回 RGB：满足 Qwen3-VL 的输入规范
        image = image.convert("RGB")

        # 3. 构建对话
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]

        # 4. 模型生成 (使用动态分辨率参数)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Qwen3-VL 32B 建议设置合理的 max_pixels 以处理表格细节 
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=8192,
                do_sample=False  # 提取表格数据建议关闭随机性，使用 Greedy Search
            )
        
        # 5. 解码
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # 核心清理动作
        del inputs, generated_ids, pix, image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return output_text # 返回内容和偏移量以供后续解析坐标
    
    def compute_render_params(
        self,
        clean_doc: fitz.Document,
        target_scale=2.0,
        max_tile_pixels=1800,
        min_tile_pixels=448,
    ):  
        """
        3. 使用模型对单页 PDF 进行原生解析，返回 Markdown 文本,方式为滑窗切片接力。
        计算 PDF 渲染参数，返回 scale, tile_height, overlap, rendered_w_aligned
        1. 根据 PDF 正文字体大小和 target_scale，计算最终渲染的 scale，确保正文字体高度在 16~24px 范围内。
        2. 根据渲染后的页面宽度和面积约束，计算 tile_height，确保不超过 MAX_PIXELS。
        3. 根据 tile_height 计算 overlap 和 stride，确保 tile 间有适当重叠。
        4. 返回最终的渲染参数，供后续裁剪和 VLM 识别使用。
        """
        MAX_PIXELS = 1280 * 28 * 28  * 2 # 2,007,040，Qwen3-VL 默认面积上限
        
        def align_to_28(pixels: float) -> int:
            return round(pixels / 28) * 28

        def floor_to_28(pixels: float) -> int:
            """面积约束时必须向下取整，不能超。"""
            return (int(pixels) // 28) * 28

        # 入口对齐边界参数
        min_tile_pixels = align_to_28(min_tile_pixels)
        max_tile_pixels = align_to_28(max_tile_pixels)

        # 一次性打开 PDF
        doc = clean_doc
        page = doc[0]
        page_w = page.rect.width
        page_h = page.rect.height

        font_sizes = []
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_sizes.append(span["size"])

        body_font_size = (
            Counter(font_sizes).most_common(1)[0][0] if font_sizes else 11.0
        )
        target_char_height = body_font_size * target_scale
        if not (16 <= target_char_height <= 24):
            # 超出范围则反推合适的 scale
            if target_char_height < 16:
                target_scale = round(16 / body_font_size, 1)
            else:
                target_scale = round(24 / body_font_size, 1)
            target_char_height = body_font_size * target_scale

        scale = target_char_height / body_font_size
        scale = round(max(1.0, min(scale, 3.0)), 1)

        rendered_w = page_w * scale
        rendered_h = page_h * scale

        # 宽度对齐（渲染时也用这个值）
        rendered_w_aligned = align_to_28(rendered_w)

        # ── tile_height：三重约束取最严 ─────────────────────────────────────
        # 1. 用户指定的高度上限
        h_from_param = max_tile_pixels

        # 2. 面积约束反解（向下取整，绝对不能超 MAX_PIXELS）
        h_from_area = floor_to_28(MAX_PIXELS / rendered_w_aligned)

        # 3. 页面实际渲染高度（tile 不必超过整页）
        h_from_page = floor_to_28(rendered_h)

        tile_height = min(h_from_param, h_from_area, h_from_page)
        tile_height = max(tile_height, min_tile_pixels)

        # 面积约束比 min_tile_pixels 还严时，发出警告（scale 过大）
        if h_from_area < min_tile_pixels:
            actual_area = rendered_w_aligned * min_tile_pixels
            logging.info(
                f"⚠️  警告：当前 scale={scale} 导致页面宽度 {rendered_w_aligned}px，"
                f"min_tile={min_tile_pixels}px 时面积 {actual_area:,} > MAX_PIXELS {MAX_PIXELS:,}。"
                f"建议降低 scale 或 min_tile_pixels。"
            )

        # ── overlap & stride ─────────────────────────────────────────────────
        overlap = align_to_28(tile_height * 0.175)
        max_overlap = (tile_height // 2 // 28) * 28
        overlap = min(overlap, max_overlap)
        if overlap == 0 and tile_height > 56:
            overlap = 28

        stride = tile_height - overlap

        logging.info(f"📐 Qwen3-VL 优化自动渲染参数:")
        logging.info(f"   正文字体 = {body_font_size:.1f}pt → target_char_height={target_char_height:.1f}px → scale={scale}")
        logging.info(f"   页面渲染尺寸 = {rendered_w_aligned} × {int(rendered_h)} px")
        logging.info(f"   面积约束 tile_height 上限 = {h_from_area} (MAX_PIXELS / {rendered_w_aligned})")
        logging.info(
            f"   tile_height = {tile_height} (28×{tile_height // 28}) | "
            f"overlap = {overlap} (28×{overlap // 28}) | "
            f"stride = {stride}"
        )
        if stride <= 0:
            raise ValueError(
                f"计算得到的 stride 非法 (stride={stride} <= 0)："
                f"tile_height={tile_height}, overlap={overlap}。"
                f"请检查 max_tile_pixels/min_tile_pixels 参数是否合理。"
            )

        return scale, tile_height, overlap, rendered_w_aligned
    
    def _run_vlm_stage(self, clean_doc: fitz.Document, user_prompt, tile_height=2000, overlap=350, scale=2.0, rendered_w_aligned=0, vlm='Qwen--Qwen3-VL-32B-Instruct') -> str:
        """
        3. 使用模型对单页 PDF 进行原生解析，返回 Markdown 文本,方式为滑窗切片接力。
        第一阶段：VLM 提取。保持窄长比例不引入白边，依赖模型原生动态分辨率看清细节
        """
        vl_model, vl_processor = self.get_vlm_model(vlm)
        logging.info("🔍 [VLM Stage] 正在处理 PDF 页面并构建滑窗输入...")
        
        doc = clean_doc
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # 若传入了对齐宽度，裁掉右侧多余像素使宽度精确对齐 28 的倍数
        if rendered_w_aligned is not None and image.width != rendered_w_aligned:
            image = image.crop((0, 0, rendered_w_aligned, image.height))

        # 图像预处理
        image = image.convert("L")
        image = ImageOps.autocontrast(image, cutoff=0.5)
        image = image.convert("RGB")
        w, h = image.size
        
        # ─── 🚀 【核心前置增强：物理长图动态垫白底】 ──────────────────────────
        stride = tile_height - overlap
        
        # 1. 掐指一算：看看不往前拽起点的情况下，总共需要切几块才能覆盖全图
        import math
        # 除去第一个标准块占用的高度，剩下所需的净高度除以有效步长，向上取整再加1
        num_tiles = math.ceil((h - tile_height) / stride) + 1 if h > tile_height else 1
        
        # 2. 计算这 num_tiles 个标准块在不发生“起点前拽”时，需要的绝对完美总高度
        perfect_total_height = (num_tiles - 1) * stride + tile_height
        
        # 3. 如果当前原图高度小于完美高度，说明最后一块如果不前挪就会“露底”。立刻向下长出纯白边！
        if h < perfect_total_height:
            pad_needed = perfect_total_height - h
            logging.info(f"➕ [Padding] 监测到最后一块跨度断层，正在向下补白 {pad_needed}px 形成几何闭环...")
            padded_image = Image.new("RGB", (w, perfect_total_height), (255, 255, 255))
            padded_image.paste(image, (0, 0))
            image = padded_image
            w, h = image.size # 更新画布的全局高宽，此时 h 已经是 perfect_total_height 了！

            logging.info(f"✅ 补白完成，新的图像尺寸: {w}×{h}px，完美闭环覆盖 {num_tiles} 个标准切片。")
        # ─── 1. 标准 A4 滑动窗口切片（无白边、高密度接力） ──────────────────────────

        tiles = []
        tile_offsets = []  # 记录每一个 tile 真实的 Y 轴像素起点
        start_y = 0
        stride = tile_height - overlap
        
        while start_y < h:
            end_y = start_y + tile_height
            current_start_y = start_y
            
            if end_y >= h:
                end_y = h
                current_start_y = max(0, h - tile_height)
            
            # 截取标准 A4 比例区域
            tiles.append(image.crop((0, current_start_y, w, end_y)))
            tile_offsets.append(current_start_y)  # 精准记录绝对 Y 偏移
            
            if end_y == h: 
                break
            start_y += stride

        # # ─── 🚀 【新增：切片图像持久化保存功能】 ──────────────────────────
        # save_dir = "/workspace/hf-conda/RAG/问答机器人/other/finebi/函数专题/1_函数新手入门/"
        # os.makedirs(save_dir, exist_ok=True)
        
        # for idx, (t_img, t_offset) in enumerate(zip(tiles, tile_offsets)):
        #     save_path = os.path.join(save_dir, f"tile_{idx}_y{t_offset}.png")
        #     t_img.save(save_path, "PNG")
            
        # logging.info(f"💾 已成功将 {len(tiles)} 个切片图像保存至目录: {save_dir}")
        # # ────────────────────────────────────────────────────────────────


        logging.info(f"📄 原始切片完成: {len(tiles)} 个tile，原图高宽比: {h/w:.2f}，start_y={start_y}, stride={stride}, tile_height={tile_height}, overlap={overlap}")

        vlm_raw_outputs = []
        last_heading_context = "无（这是第一块，请直接开始解析）"
        tile_output_global = ''

        # logging.info_mem("循环开始前")
        for idx in range(len(tiles)):
            # logging.info_mem(f"tile {idx} 推理前")
            current_start_y = idx * stride
            if idx == len(tiles) - 1:
                current_start_y = max(0, h - tile_height)
                
            logging.info(f" 正在处理第 {idx+1}/{len(tiles)} 个切片...")
            current_input_images = []
            content_list = []
            
            # 拿到100%高密度、无空白污染的窄长局部切片
            current_tile = tiles[idx]
            tile_w, tile_h = current_tile.size

            # ─── 2. 动态 Prompt 构建（不变） ──────────────────────────────────────────
            is_first_tile = (idx == 0)
            instructions = ["━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n【当前切片动态上下文与状态分流宣告】\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"]
            if is_first_tile:
                instructions.append(
                    f"⚠️【当前切片状态：局部切片感知模式（最高红线）】\n"
                    f"- **坐标纯净命令：** 请完全忽略任何长图概念。你当前看到的图片高为 {tile_h}px，宽为 {tile_w}px。\n"
                    f"- **垂直坐标有效区间：** 你输出的 `<box>(xmin, ymin, xmax, ymax)</box>` 中，\n"
                    f"  `ymin` 和 `ymax` 必须**严格限制在 0 到 {tile_h} 之间**（顶边为0，底边为{tile_h}）。\n"
                    f"  ⚠️ 铁律：严禁尝试自行累加任何长图偏移量！只输出当前画面内真实的局部像素位置。\n"
                    f"- **⚠️ 视觉并列标题校准令（防止并列标题误降级）：**\n"
                    f"  当前图片为整个文档的开篇第一块。请注意，单张图片内可能存在**多个相互独立、级别完全并列的一级标题**。\n"
                    f"  你必须纯粹依据【视觉字号大小和粗细】来判定层级。只要它们字号完全相同、粗细完全一致、且均处于独立大章节地位，**必须全部统一判定为一级标题 `#`**。绝对禁止因为它们在同一个切片内先后出现，就主观将靠后的并列大标题自动顺延降级为 `##`！"
                )

            else:
                current_tile_offset = current_start_y
                instructions.append(
                    f"⚠️【当前切片状态：中间延续切片 (切片空间警戒线：0 ~ {tile_h} px）】\n"
                    f"- **分流执行指令：** 当前切片属于长文档的深度延续部分（Tile_start = {current_tile_offset}px）。\n"
                    f"- **坐标纯净死命令：** \n"
                    f"  请你将大脑完全聚焦于当前这一张图片空间。当前图片内出现的所有新元素或跨页延续表格（R4），其坐标 `<box>(xmin, ymin, xmax, ymax)</box>` **必须严格输出当前单张图片内的局部相对像素！**\n"
                    f"  垂直坐标（ymin/ymax）的有效范围必须死死限制在 `0` 到 `{tile_h}` 像素之间（顶边为0，底边为{tile_h}）。\n"
                    f"  ⚠️ 铁律：严禁在输出中自行加上任何长图偏移量底数，不需要你做加法！请纯粹输出你在这张切片上肉眼看到的像素位置。"
                    f"- **规则激活限制：** 你必须【全量激活】主 Prompt 第二节的滑动窗口接力规则，并死守优先级铁律：`R4 > R1 > R3 > R2 > R5 > R6`。\n"
                )

                # R1 规则的动态变量精准注入
                instructions.append(
                    f"💡【R1 规则动态上下文变量注入】\n"
                    f"- 经系统判定，当前前文所处的最底层章节大纲基准为：`{last_heading_context}`。\n"
                    f"- **骨架继承应用：** 若当前图片顶部出现的标题命中了该基准，请严格按照 R1 规则在开头声明该标题骨架，绝不允许子标题或正文“空降”导致目录树断裂。\n"
                )

                # R3 / R4 规则在当前重叠区的核验协同指引
                instructions.append(
                    f"💡【R3 / R4 规则重叠区协同核验指引】\n"
                    f"- **第一步：延续核验（R4触发）。** 优先检查顶部约 {overlap}px 重叠区，若上文末尾的表格在此处有物理线条延续或符合主 Prompt 步骤④的对齐延续，强制触发最高优先级 R4。按 R1->R4 顺序输出，ymax 执行物理截断阻断（绝不允许包裹下方无关段落）。\n"
                    f"- **第二步：残缺核验（R3触发）。** 若未命中表格延续，且检查发现上一个切片的参考文本末尾行存在话未说完、字形切碎、括号不匹配等物理切断痕迹，强制触发 R3 规则。必须在当前输出流第一行完美补出该残缺元素的剩余文本，实现无缝拼接。\n"
                    f"- **第三步：硬去重（R2触发）。** 排除上述接力、延续情况后，凡是在参考文本中已完整出现的普通段落、列表项，当前切片一律执行 R2 熔断（闭嘴并禁止输出坐标）。\n"
                )
            user_prompt_filled = user_prompt.replace("{overlap}", str(int(overlap)))
            
            step_prompt = f"{user_prompt_filled}\n---\n" + "\n".join(instructions)

            # ─── 3. 输入构建与推理 ──────────────────────────────────────────
            content_list.append({"type": "text", "text": f"[解析对象：当前第 {idx+1} 块画面]"})
            content_list.append({"type": "image", "image": current_tile})
            current_input_images.append(current_tile)
            content_list.append({"type": "text", "text": step_prompt})

            messages = [{"role": "user", "content": content_list}]
            text = vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # 🔥🔥 关键修正：通过配置让 Processor 针对窄长图智能分配最佳 Patch，绝不进行压扁缩放
            inputs = vl_processor(
                text=[text], 
                images=current_input_images, 
                padding=True, 
                return_tensors="pt"
            ).to(vl_model.device)

            with torch.no_grad():
                # 配合 do_sample=False 稳定输出，适当降低惩罚防止表格标签被截断
                generated_ids = vl_model.generate(**inputs, max_new_tokens=4096, do_sample=False, repetition_penalty=1.1, no_repeat_ngram_size=10)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            tile_output = vl_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            

            # ─── 4. 状态机更新与全局 Y 轴还原（X轴不再需要缩放因子） ─────────────────────────

            
            headings = re.findall(r'^(#{1,6}\s+.*?<box>.*?</box>)', tile_output_global, re.MULTILINE)
            if headings:
                raw_heading = headings[-1]
                # 注入前剥离坐标，只保留标题文本
                clean_heading = re.sub(r'<box>.*?</box>', '', raw_heading).strip()
                # 额外防御：清理任何畸形 box 残留（如 <box) 等）
                clean_heading = re.sub(r'<box[^>]*>|</box[^>]*>', '', clean_heading).strip()
                last_heading_context = clean_heading
            else:
                pass
            # 获取当前tile的实际尺寸（像素/点数）
            tile_w_actual, tile_h_actual = current_tile.size

            def single_column_adaptive_restore(m, th, tw, start_y, scale_factor):
                # 1. 拿到原始 4 个数值
                ymin_raw = float(m.group(2))
                ymax_raw = float(m.group(4))

                # 防御：确保 min < max（偶发的坐标倒置）
                ymin_raw, ymax_raw = sorted([ymin_raw, ymax_raw])

                # 3. 【核心修正公式】
                # 将 Tile 的像素尺寸先除以 scale，转换为标准 PDF 的 pt 尺寸，再进行比例映射
                tile_w_pt = tw / scale_factor
                tile_h_pt = th / scale_factor
                tile_start_pt = start_y / scale_factor  # 确保起点也是 pt 单位
                
                real_ymin = round(tile_start_pt + (ymin_raw * tile_h_pt / 1000.0), 1)
                real_ymax = round(tile_start_pt + (ymax_raw * tile_h_pt / 1000.0), 1)

                # 5. X 轴标准通栏输出
                real_xmin = 0.0
                real_xmax = round(tile_w_pt, 1)

                return real_xmin, real_ymin, real_xmax, real_ymax
            
            box_pattern = re.compile(
                r"<b[^>]*?>\s*[\D]*?\s*"       
                r"(-?\d+(?:\.\d+)?)\s*,\s*"    
                r"(-?\d+(?:\.\d+)?)\s*,\s*"    
                r"(-?\d+(?:\.\d+)?)\s*,\s*"    
                r"(-?\d+(?:\.\d+)?)"           
                r"\s*[\D]*?\s*"                
                r"</b[^>]*?>",                 
                re.IGNORECASE
            )
           
            # 🎯 【更稳固的显式回调函数，绝不漏掉任何一个 Tile 的坐标】
            def _box_replacer(m, tw=tile_w_actual, th=tile_h_actual, sy=current_start_y, sc=scale):
                # 1. 运算出全局真实的 pt 坐标
                real_xmin, real_ymin, real_xmax, real_ymax = single_column_adaptive_restore(
                    m, th=th, tw=tw, start_y=sy, scale_factor=sc
                )
                # 2. 🎯 必须返回一个标准的字符串，格式化为新标签替换回文本中！
                return f"<box>({real_xmin:.1f}, {real_ymin:.1f}, {real_xmax:.1f}, {real_ymax:.1f})</box>"
            
            # 你原本的替换代码
            tile_output_global = box_pattern.sub(_box_replacer, tile_output)
            
            vlm_raw_outputs.append(f"\n\n<!-- tile_start={current_start_y} -->\n{tile_output_global}\n")
            logging.info(f"==================================================")
            del inputs, generated_ids, current_input_images, text, tile_output, current_tile
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ─── 5. 后置全局垃圾回收与物理排序（已移至循环外） ──────────────────────────────────
        del tiles, image

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        raw_document_draft = "\n\n".join(vlm_raw_outputs)

        logging.info("🖼️ 正在基于全局坐标对所有图片占位符进行物理排序...")

        # 1. 使用命名捕获组的正则表达式
        image_placeholder_pattern = re.compile(
            r"!\[IMAGE\]"
            r"(?:\([^\)]*\))?"
            r"(?:(?!!\[IMAGE\])[^\d\n])*?"
            r"(?P<xmin>\d+(?:\.\d+)?)"
            r"[\s,\)\(]+"
            r"(?P<ymin>\d+(?:\.\d+)?)"
            r"[\s,\)\(]+"
            r"(?P<xmax>\d+(?:\.\d+)?)"
            r"[\s,\)\(]+"
            r"(?P<ymax>\d+(?:\.\d+)?)"
            r"\)?\s*</box>(?:\)?\s*</box>)?",   # 新增：吃掉收尾的 )、</box>，包括可能重复的一层
            re.IGNORECASE
        )

        # 2. 提取所有匹配项对象 (finditer 比 findall 能拿到具名字典，更安全)
        matches = list(re.finditer(image_placeholder_pattern, raw_document_draft))
        # 3. 提取唯一坐标并标准化 (转换为 float tuple 消除字符串格式差异)
        unique_coords = set()
        for m in matches:
            coord_tuple = (
                float(m.group("xmin")),
                float(m.group("ymin")),
                float(m.group("xmax")),
                float(m.group("ymax"))
            )
            unique_coords.add(coord_tuple)

        # 4. 科学排序：按照文档阅读顺序 (先按 ymin 上到下，再按 xmin 左到右)
        sorted_unique_images = sorted(unique_coords, key=lambda c: (c[1], c[0]))

        # 5. 构建稳定映射字典 (Key 直接使用 float 元组)
        image_coord_to_name_map = {
            coord: f"image_{i}" for i, coord in enumerate(sorted_unique_images, start=1)
        }
        
        # 6. 替换回调函数
        def rename_image_placeholders(match):
            # 转换为统一的 float tuple 进行 Key 查找
            current_coord = (
                float(match.group("xmin")),
                float(match.group("ymin")),
                float(match.group("xmax")),
                float(match.group("ymax"))
            )
            
            final_img_name = image_coord_to_name_map.get(current_coord, "image_unknown")
            
            # 格式化输出统一标准的坐标串
            xmin_str, ymin_str, xmax_str, ymax_str = match.group("xmin"), match.group("ymin"), match.group("xmax"), match.group("ymax")
            return f"![IMAGE](IMAGE_PLACEHOLDER) <box>({xmin_str}, {ymin_str}, {xmax_str}, {ymax_str})</box>"

        # 7. 执行替换
        ordered_document_draft = re.sub(image_placeholder_pattern, rename_image_placeholders, raw_document_draft)
        # ─── 🚀 【物理空间锁 + Levenshtein相似度 复合流式去重管道】 ──────────────────────────
        PLACEHOLDER_TAG_RE = re.compile(
            r'^\s*(\[(TABLE_PLACEHOLDER|IMAGE_PLACEHOLDER|TEXT_BLOCK_CONTINUE)\]'
            r'|!\[IMAGE\]\(IMAGE_PLACEHOLDER\))',
            re.IGNORECASE
        )


        def normalize_line(line: str) -> str:
            """
            归一化一行文本，用于内容级别的去重比对：
            - 全角转半角（字母、数字、标点、空格）
            - 大小写统一转小写
            - 连续空白（含全角空格、制表符）压缩为单个空格
            - 去除首尾空白
            - 去掉行内的坐标 box 标记（避免坐标抖动影响文本比对）
            """
            if not line:
                return ''

            # 1. 去掉 <box>...</box>，因为坐标不参与内容比对
            text = re.sub(r'<box>.*?</box>', '', line)

            # 2. 全角转半角（NFKC 会把全角字母数字标点、全角空格 都转成对应半角形式）
            text = unicodedata.normalize('NFKC', text)

            # 3. 大小写统一
            text = text.lower()

            # 4. 压缩连续空白为单个空格（NFKC 之后全角空格已经变成普通空格，这里一并处理制表符等）
            text = re.sub(r'\s+', ' ', text)

            # 5. 去首尾空白
            text = text.strip()

            return text

        BOX_RE = re.compile(r'<box>\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)</box>')
        TILE_START_RE = re.compile(r'<!--\s*tile_start=(\d+)\s*-->')

        def dedupe_by_tile_ymax(ordered_document_draft: str,
                                keep_tile_markers: bool = False,
                                recent_window: int = 30) -> str:
            """
            按 tile 滑窗合并去重，双重判断：
            1) 坐标水位线：ymax 必须严格大于此前保留内容的最大 ymax；
            2) 内容归一化去重：即使 ymax 判断通过，如果归一化文本命中
            "最近保留过的内容"缓存，也判定为重复行（大小写/全半角/多余空格
            导致的抖动不会绕过去重）。
            """
            blocks = re.split(r'(?=<!--\s*tile_start=\d+\s*-->)', ordered_document_draft)
            blocks = [b for b in blocks if b.strip() != '']

            kept_lines = []
            last_ymax = None
            recent_norm_texts = []  # 滑动窗口，存最近保留的归一化文本，避免全量比对拖慢速度

            def is_recent_duplicate(norm_text: str) -> bool:
                if not norm_text:
                    return False
                return norm_text in recent_norm_texts

            def push_recent(norm_text: str):
                if not norm_text:
                    return
                recent_norm_texts.append(norm_text)
                if len(recent_norm_texts) > recent_window:
                    recent_norm_texts.pop(0)

            for block in blocks:
                for line in block.split('\n'):
                    tile_m = TILE_START_RE.search(line)
                    if tile_m:
                        if keep_tile_markers:
                            kept_lines.append(line)
                        continue

                    box_m = BOX_RE.search(line)
                    is_placeholder = bool(PLACEHOLDER_TAG_RE.match(line))

                    if not box_m:
                        # 没有坐标的行：占位符类不做内容去重判断，其余仍做
                        if not is_placeholder:
                            norm_text = normalize_line(line)
                            if is_recent_duplicate(norm_text):
                                continue
                            push_recent(norm_text)
                        kept_lines.append(line)
                        continue

                    ymax = float(box_m.group(4))

                    # 第一层：坐标水位线判断
                    if last_ymax is not None and ymax <= last_ymax:
                        continue  # 落在已覆盖范围内，丢弃

                    # 第二层：内容归一化去重（兜底，处理坐标抖动导致的误放行）
                    if not is_placeholder:
                        norm_text = normalize_line(line)
                        if is_recent_duplicate(norm_text):
                            continue
                        push_recent(norm_text)
                    kept_lines.append(line)
                    last_ymax = ymax if last_ymax is None else max(last_ymax, ymax)

            return '\n'.join(kept_lines)

        # 用法
        
        ordered_document_draft = re.sub(image_placeholder_pattern, rename_image_placeholders, raw_document_draft)
        leftover = re.findall(r'</box>\s*\)?\s*</box>', ordered_document_draft)
        if leftover:
            logging.warning(f"⚠️ 检测到 {len(leftover)} 处疑似残留的多余 </box> 标签，image_placeholder_pattern 可能未完全覆盖")
        ordered_document_draft = dedupe_by_tile_ymax(ordered_document_draft)
        ordered_document_draft = re.sub(r'\n{3,}', '\n\n', ordered_document_draft)  # 去除多余空行
        logging.info("✅ 全局坐标排序与跨 Tile 去重完成，输出最终有序文档草稿。")
        return ordered_document_draft

    def extract_tables_as_html(self, clean_doc: fitz.Document, structure, tile_height=2000, overlap=350, scale=2.0,vlm='Qwen--Qwen3-VL-32B-Instruct') -> list:
        """
        3. 使用模型对单页 PDF 进行原生解析，返回 Markdown 文本,方式为滑窗切片接力。
        第二阶段：表格裁剪:从 structure 中提取表格坐标，修正偏移，IoU 去重后裁剪交给 VLM 识别。返回: [{"html": ..., "box": (ymin, xmin, ymax, xmax)}, ...]
        """
        # logging.info(f"🔍 [Table Extraction] 正在解析 PDF 表格坐标，tile_height={tile_height}, overlap={overlap}")
        stride = tile_height - overlap  

        # ── 局部工具函数 ──────────────────────────────────────────────────────

        def iou(a, b):
            # a, b 格式: (x1, y1, x2, y2)，但 x1=0, x2=tw
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b

            inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
            inter_h = max(0, min(ay2, by2) - max(ay1, by1))
            inter_area = inter_w * inter_h

            # 计算各自的面积
            area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
            
            union_area = area_a + area_b - inter_area
            return inter_area / union_area if union_area > 0 else 0.0


        def ioa(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b

            # 计算交集面积
            inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
            inter_h = max(0, min(ay2, by2) - max(ay1, by1))
            inter_area = inter_w * inter_h
            area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

            if area_a == 0 or area_b == 0:
                return 0.0
                
            # 返回交集占任意一个矩形面积的最大比例
            return max(inter_area / area_a, inter_area / area_b)
        
        
        def merge_cross_tile(entries, structure, stride, overlap, scale):
            """
            跨 Tile 的表格坐标缝合。对同一表格在不同 tile 中的坐标进行 IoU/IoA 检测，若满足条件则合并为一个全局坐标。
            1. 解析所有 IMAGE 坐标，确保表格不会被截断到图片中。
            2. 对于同一表格在不同 tile 中的坐标，若满足 IoU/IoA 阈值，则进行合并，确保表格的完整性。
            3. 对于合并后的坐标，若存在 IMAGE 截断的情况，则将 ymax 截断到 IMAGE 的 ymin - margin，确保表格不会被截断到图片中。
            4. 返回合并后的全局坐标列表
            """
            if not entries:
                return entries
        
            overlap_pt          = overlap / scale
            tile_height_pt      = (stride + overlap) / scale
            y_tolerance_pt      = overlap_pt / 2.0
            adjacent_y_tol_pt   = max(y_tolerance_pt, tile_height_pt * 0.3)
            x_overlap_threshold = 0.3
            IMAGE_MARGIN_PT     = 8.0   # 截断点距 IMAGE 顶边的安全边距
        
            # ── 解析所有 IMAGE 坐标 ──────────────────────────────────────
            image_pat = re.compile(
                r"!\[IMAGE\]"                                  # 1. 匹配 ![IMAGE]
                r"(?:\([^\)]*\))?"                             # 2. 匹配可选的 (IMAGE_PLACEHOLDER) 或 (image_1)
                r"(?:(?!!\[IMAGE\])[^\d\n])*?"                 # 3. 跨过非数字字符，但绝不允许跨过下一个 ![IMAGE]
                r"(?P<xmin>\d+(?:\.\d+)?)"
                r"[\s,\)\(]+"
                r"(?P<ymin>\d+(?:\.\d+)?)"
                r"[\s,\)\(]+"
                r"(?P<xmax>\d+(?:\.\d+)?)"
                r"[\s,\)\(]+"
                r"(?P<ymax>\d+(?:\.\d+)?)",
                re.IGNORECASE
            )
            image_ranges = [
                (float(m.group(2)), float(m.group(4)))   # (ymin, ymax)
                for m in image_pat.finditer(structure)
            ]
        
            def clip_ymax_by_images(ymin, ymax, keeper_original_ymax):
                """
                如果 ymin~ymax 范围内有 IMAGE，把 ymax 截断到：
                max(keeper_original_ymax, first_image_ymin - IMAGE_MARGIN_PT)
                确保上段真实内容被包含，UI 截图被排除。
                """
                overlapping = [
                    img_ymin for img_ymin, img_ymax in image_ranges
                    if img_ymin > ymin + 20.0   # 排除表格顶部微小重叠
                    and img_ymin < ymax
                ]
                if not overlapping:
                    return ymax, False   # 无截断
        
                first_img_ymin = min(overlapping)
                clipped = max(keeper_original_ymax, first_img_ymin - IMAGE_MARGIN_PT)
                return clipped, True
            # ─────────────────────────────────────────────────────────────
        
            sorted_entries = sorted(entries, key=lambda x: x['box'][1])
            merged_results = []
        
            for candidate in sorted_entries:
                c_xmin, c_ymin, c_xmax, c_ymax = candidate['box']
                c_start = candidate['tile_start']
                has_merged = False
        
                for keeper in reversed(merged_results):
                    k_xmin, k_ymin, k_xmax, k_ymax = keeper['box']
                    k_start = keeper['tile_start']
        
                    if c_start == k_start or abs(c_start - k_start) > (stride + overlap):
                        continue
        
                    inter_w = max(0.0, min(c_xmax, k_xmax) - max(c_xmin, k_xmin))
                    min_w   = min(c_xmax - c_xmin, k_xmax - k_xmin)
                    x_ratio = inter_w / min_w if min_w > 0 else 0
        
                    y_gap = c_ymin - k_ymax
                    is_adj = abs(c_start - k_start) == stride
        
                    if is_adj:
                        max_gap_error = min(y_tolerance_pt, 15.0)
                        is_y_ok = (y_gap < 0 and abs(y_gap) <= overlap_pt + adjacent_y_tol_pt) \
                                or (0 <= y_gap <= max_gap_error)
                    else:
                        is_y_ok = (y_gap < 0 and abs(y_gap) <= overlap_pt + y_tolerance_pt) \
                                or (0 <= y_gap <= y_tolerance_pt)
        
                    if not (x_ratio >= x_overlap_threshold and is_y_ok):
                        continue
        
                    # ── 几何条件通过，计算合并后的原始 box ────────────────
                    new_xmin = min(k_xmin, c_xmin)
                    new_ymin = min(k_ymin, c_ymin)
                    new_xmax = max(k_xmax, c_xmax)
                    new_ymax = max(k_ymax, c_ymax)
                    
        
                    # ── IMAGE 截断检查（核心新增）─────────────────────────
                    clipped_ymax, was_clipped = clip_ymax_by_images(
                        new_ymin, new_ymax,
                        keeper_original_ymax=k_ymax   # 上段原始 ymax 作为下限
                    )
    
                    keeper['box'] = (new_xmin, new_ymin, new_xmax, clipped_ymax)
                    keeper['tile_start'] = c_start
        
                    tag = "严格相邻-宽松带" if is_adj else "常规阈值"
                    logging.info(f"  🤝 [merge_cross_tile] 缝合 [{tag}]: {k_start}px → {c_start}px")
                    logging.info(f"     ├─ 上方: {k_ymin:.1f}~{k_ymax:.1f}pt")
                    logging.info(f"     ├─ 下方: {c_ymin:.1f}~{c_ymax:.1f}pt "
                        f"(Gap:{y_gap:.1f}pt X:{x_ratio:.2f})")
                    if was_clipped:
                        logging.info(f"     ├─ ✂️  IMAGE 截断: ymax {new_ymax:.1f}→{clipped_ymax:.1f}pt")
                    logging.info(f"     └─ 结果: {new_ymin:.1f}~{clipped_ymax:.1f}pt")
        
                    has_merged = True
                    break
        
                if not has_merged:
                    merged_results.append({
                        "box": candidate["box"],
                        "tile_start": c_start,
                        **{k: v for k, v in candidate.items() if k not in ["box", "tile_start"]}
                    })
        
            return merged_results

        # 🎯 【终极进化：无视任何幻觉标签的通用数字提取正则】
        table_box_pat = re.compile(
            r"\[(TABLE_PLACEHOLDER|MIXED_TABLE_PLACEHOLDER)\]"   # group(1): 匹配占位符
            r"(?:\s*\[[^\]]*\])?"                                # 优雅吞掉内联描述
            r".*?"                                               # 核心跳跃：用非贪婪通配符，直接跨过任何脑补的 <box>、<bbox>、<boxed> 标签
            r"\(\s*"                                             # 紧紧咬住左括号 (
            r"(-?\d+(?:\.\d+)?)\s*,\s*"                          # group(2): 第1个数字 (xmin)
            r"(-?\d+(?:\.\d+)?)\s*,\s*"                          # group(3): 第2个数字 (ymin)
            r"(-?\d+(?:\.\d+)?)\s*,\s*"                          # group(4): 第3个数字 (xmax)
            r"(-?\d+(?:\.\d+)?)"                                 # group(5): 第4个数字 (ymax)
            r"\s*\)"                                             # 紧紧咬住右括号 )
            r"(?:.*?</.*?>)?",                                   # 可选：兼容后面可能有的任意闭合标签
            re.IGNORECASE                                        
        )

        tile_start_pat = re.compile(r"<!--\s*tile_start=(\d+)\s*-->")

        entries = []
        current_tile_start = 0

        for line in structure.splitlines():
            ts = tile_start_pat.search(line)
            if ts:
                current_tile_start = int(ts.group(1))

            for m in table_box_pat.finditer(line):   # ← search 改 finditer
                x1, y1, x2, y2 = (float(m.group(i)) for i in range(2, 6))
                if x1 >= x2 or y1 >= y2 or y1 < 0 or x1 < 0:
                    logging.info(f"⚠️  丢弃无效 box: ({x1},{y1},{x2},{y2})")
                    continue
                else:
                    entries.append({
                        "box": (x1, y1, x2, y2),
                        "tile_start": current_tile_start,
                    })
                    logging.info(f"  📌 收集: box=({x1},{y1},{x2},{y2}) tile_start={current_tile_start}")
            
        if not entries:
            logging.info("⚠️ structure 中未找到 TABLE_PLACEHOLDER")
            return []

        # ── step2: 跨 tile 合并 ─────────────────────────────────────────────
        entries = merge_cross_tile(entries, structure, stride, overlap, scale)  # ← 新增这一行
        logging.info(f"📊 跨tile合并后={len(entries)}")
        for e in entries:
            logging.info(f"    box={e['box']}")  # 新增

        # ── step3: IoU/IoA 去重，保留较早 tile ───────────────────────────────
        entries = sorted(entries, key=lambda e: (e["box"][1], e["box"][0]))
        before_dedup = len(entries)
        kept = []
        for candidate in entries:
            matched = False
            for k in kept:
                i = iou(candidate["box"], k["box"])
                a = ioa(candidate["box"], k["box"])
                if i >= 0.5 or a >= 0.4:
                    logging.info(f"  🗑️  去重: {candidate['box']} ← 被 {k['box']} 覆盖 (iou={i:.2f} ioa={a:.2f})")
                    matched = True
                    break
            if not matched:
                logging.info(f"  ✅  保留: {candidate['box']}")
                kept.append(candidate)


        logging.info(f"📊 TABLE_PLACEHOLDER 合并后={before_dedup} 去重后={len(kept)}")
        for e in kept:
            x1, y1, x2, y2 = e["box"]
            logging.info(f"box={e['box']} → x1={x1} y1={y1} x2={x2} y2={y2} → pdf x0={x1/scale:.1f} y0={y1/scale:.1f}")


        # ── step4: 裁剪 + VLM 识别 ────────────────────────────────────────────

        doc    = clean_doc
        page   = doc[0]
        page_w = page.rect.width
        page_h = page.rect.height

        vl_model, vl_processor = self.get_vlm_model(vlm)

        table_special_prompt = self.prompts.get("table_special_prompt_single_column")

        results = []
        # 🎯 设定裁剪外扩缓冲垫（单位: pt），防止表格外边框被切碎
        crop_padding = 8.0

        for idx, e in enumerate(kept):
            xmin, ymin, xmax, ymax = e["box"]   # (xmin, ymin, xmax, ymax)
            logging.info(f"  🔍 识别第 {idx+1}/{len(kept)} 个表格，box={e['box']}")

            x0_pt = max(0.0, (xmin) - crop_padding)
            y0_pt = max(0.0, (ymin) - crop_padding)
            x1_pt = min(page_w, (xmax) + crop_padding)
            y1_pt = min(page_h, (ymax) + crop_padding)
            pdf_rect = fitz.Rect(x0_pt, y0_pt, x1_pt, y1_pt)
            # logging.info(f"     📐 转换到 PDF 绝对坐标 (pt): x0={x0_pt:.1f}, y0={y0_pt:.1f}, x1={x1_pt:.1f}, y1={y1_pt:.1f}")
            # logging.info(f"     ✂️ 最终物理裁剪矩形: {pdf_rect}")

            # 1. 🎯 先把这一轮裁剪对应的 scaled 像素坐标算好，严格遵循你确认的 [xmin, ymin, xmax, ymax] 顺序！
            final_scaled_box = (int(x0_pt * scale), int(y0_pt * scale), int(x1_pt * scale), int(y1_pt * scale))

            # ── 验证矩形有效性 ──────────────────────────────────────
            if pdf_rect.is_empty or pdf_rect.width < 10 or pdf_rect.height < 10:
                logging.info(f"  ⚠️ 无效裁剪区域，跳过 box={e['box']} pdf_rect={pdf_rect}")
                results.append({
                    "html": None,
                    "flat_html": None,
                    "box": final_scaled_box,  # 🎯 统一用算好的标准 XY 格式
                    "skip": True,
                })
                continue

            
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=pdf_rect)
            logging.info(f"  📏 实际pixmap尺寸: {pix.width}×{pix.height}")  # 加这行
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            logging.info(f"  🖼️ PIL图像尺寸: {img.width}×{img.height}")     # 加这行

            # ── 新增：验证图片尺寸 ────────────────────────────────────────
            if img.width < 10 or img.height < 10:
                logging.info(f"  ⚠️ 裁剪图片过小 {img.width}×{img.height}，跳过")
                results.append({
                    "html": None,
                    "flat_html": None,
                    "box": final_scaled_box,
                    "skip": True,
                })
                continue

            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": table_special_prompt},
            ]}]
            text_input = vl_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = vl_processor(
                text=[text_input], images=[img], return_tensors="pt"
            ).to(vl_model.device)
            logging.info(f"  🤖 开始VLM推理...") 
            with torch.no_grad():
                generated_ids = vl_model.generate(
                    **inputs, max_new_tokens=2048, do_sample=False
                )

            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
            output  = vl_processor.batch_decode(
                trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            logging.info(f"  ✅ VLM推理完成，输出长度: {len(output)}")
            # del inputs, generated_ids, img
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()

            
            html_match = re.search(r"```html\s*(.*?)```", output, re.DOTALL)
            
            if html_match:
                html_content = html_match.group(1).strip()
                html_content = re.sub(r"<box>\s*\([^)]*\)\s*</box>\s*$", "", html_content, flags=re.DOTALL).strip()
                if re.search(r"<input|<button|<select|\[NOT_A_TABLE\]", output, re.IGNORECASE):
                    logging.info(f"  ⚠️ 检测到UI控件/NOT_A_TABLE标记，疑似截图误判，跳过 box={e['box']}")
                    results.append({
                        "html": None,
                        "flat_html": None,
                        "box": (int(pdf_rect.y0*scale), int(pdf_rect.x0*scale), int(pdf_rect.y1*scale), int(pdf_rect.x1*scale)),
                        "skip": True,
                    })
                    continue
                # 2. 🌟 核心修正点：在这里插入我们的 HTML 智能清洗与 JSON 转换算法
                try:
                    # 注意：这里调用我们刚才修正的函数，直接把 HTML 变成干净的 JSON 字符串
                    # 如果你希望 results 里面直接存 python 列表/字典，可以用 json.loads(rag_json) 转一下
                    flat_html = MarkdownProcessor.process_html_to_flat_html(html_content)
                except Exception as ex:
                    logging.info(f"  ❌ HTML 解析/智能填充失败: {ex}，将保留原始 HTML")
                    flat_html = html_content
                results.append({
                        "html": html_content,
                        "flat_html": flat_html,
                        "box": (
                            int(pdf_rect.x0 * scale),
                            int(pdf_rect.y0 * scale),
                            int(pdf_rect.x1 * scale),
                            int(pdf_rect.y1 * scale),
                        ),
                    })
                logging.info(f"裁减后的表格区域 (scaled): ({int(pdf_rect.x0 * scale)}, {int(pdf_rect.y0 * scale)}, {int(pdf_rect.x1 * scale)}, {int(pdf_rect.y1 * scale)})")
                
            else:
                logging.info(f"  ⚠️ VLM 未输出合法 HTML，跳过 box={e['box']}")
                results.append({
                    "html": None,
                    "flat_html": None,
                    "box": (int(pdf_rect.x0*scale), int(pdf_rect.y0*scale), int(pdf_rect.x1*scale), int(pdf_rect.y1*scale)),
                    "skip": True,
                })

        # ── 销毁 VLM ──────────────────────────────────────────────────────────
        # 1. 优先通过 factory 获取并销毁，或者直接调用统一封装的 destroy 方法
        if hasattr(self, 'factory') and self.factory is not None:
            self.factory.destroy_vlm_model()

        # 2. 深度显存垃圾回收
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            logging.info(f"💾 [表格VLM清场] allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

        results.sort(key=lambda x: x["box"][1])
        return results

    def generate_final_clean_markdown(self, clean_doc, structure, table_special, pdf_path):
        """
        [终极重构完整版]
        1. 参数完全对齐外部调用：(clean_doc, structure, table_special, pdf_path)
        2. 纯净 VLM 线性流线：彻底告别 full_text，从根本上杜绝表格内容倒灌和重叠残影。
        3. 修复硬伤：校正了标准的 Markdown 图片语法，补齐了 re/os 依赖。
        """
        # 直接复用外部已经打开的 doc 对象，规避重复磁盘读取
        doc = clean_doc 
        pdf_name = os.path.basename(pdf_path)
        # 获取唯一的第 0 页及其尺寸
        page = doc[0]
        page_rect = page.rect
        # 图片保存本地路径：output_images/<pdf文件名>/
        pdf_dir = os.path.dirname(pdf_path)
        pdf_stem = os.path.splitext(pdf_name)[0]
        output_img_dir = os.path.join(pdf_dir, pdf_stem)
        os.makedirs(output_img_dir, exist_ok=True)

        image_pat = re.compile(
            r"!\[IMAGE\]"
            r"(?:\((?P<desc>[^\)]*)\))?"                   # 捕获组 desc
            r"(?:(?!!\[IMAGE\])[^\d\n])*?"
            r"(?P<xmin>\d+(?:\.\d+)?)"
            r"[\s,\)\(<box/]+"
            r"(?P<ymin>\d+(?:\.\d+)?)"
            r"[\s,\)\(<box/]+"
            r"(?P<xmax>\d+(?:\.\d+)?)"
            r"[\s,\)\(<box/]+"
            r"(?P<ymax>\d+(?:\.\d+)?)",
            re.IGNORECASE
        )

        
        logging.info(f"--- 调试信息 ---")
        logging.info(f"PDF 总页数: {len(doc)}")

        # 统一将表格数据标准化为 List[Dict] 格式
        if isinstance(table_special, list):
            parsed_tables = table_special
            logging.info(f"传入的表格数量: {len(table_special)}")
        else:
            parsed_tables = self.parse_table_special(table_special)
            logging.info(f"传入的是字符串，已完成兼容性解析。")

        table_idx = 0
        lines = structure.strip().split('\n')
        final_output = []
        image_idx = 0

        for line in lines:
            line = line.strip()
            if not line: 
                continue

            # 🌟 1. 处理标题
            if line.startswith('#'):
                # 清洗掉 VLM 自带的视觉定位框标签，保留纯粹的 Markdown 标题
                title = re.sub(r'<box>.*?</box>', '', line).strip()
                final_output.append(title)

            # 🌟 2. 处理正文块（核心高 ROI 改造）
            elif '[TEXT_BLOCK]' in line:
                # 彻底放弃去物理流反查，直接提取并清洗 VLM 识别出的文本内容
                clean_text = re.sub(r'<box>.*?</box>', '', line).replace('[TEXT_BLOCK]', '').strip()
                if clean_text:
                    final_output.append(clean_text)

            # 🌟 3. 处理表格占位符
            elif '[TABLE_PLACEHOLDER]' in line:
                if table_idx < len(parsed_tables):
                    # 优先获取平铺的 flat_html（如果有的话），否则使用常规 html
                    table_content = parsed_tables[table_idx].get("flat_html") or parsed_tables[table_idx].get("html", "")
                    final_output.append(table_content.strip())
                    table_idx += 1
                else:
                    logging.info(f"⚠️ 警告: 骨架中的第 {table_idx} 个表格在解析数据中未找到（越界）")

            # 🌟 4. 处理图片占位符并裁剪保存
            elif '![IMAGE]' in line:
                PADDING = 3.0
                # print(f"\n[DEBUG] 处理图片标记行 -> 原始文本: {repr(line)}")
                match = image_pat.search(line)


                if match:
                    group_data = match.groupdict()
                    # 提取描述与坐标
                    img_desc = (group_data.get('desc') or "图片").strip()
                    xmin = float(group_data.get('xmin'))
                    ymin = float(group_data.get('ymin'))
                    xmax = float(group_data.get('xmax'))
                    ymax = float(group_data.get('ymax'))

                    # 边界外扩 Padding 防切边，并用页面真实边界做 Clamp 保护
                    page_rect = page.rect
                    crop_xmin = max(0, xmin - PADDING)
                    crop_ymin = max(0, ymin - PADDING)
                    crop_xmax = min(page_rect.width, xmax + PADDING)
                    crop_ymax = min(page_rect.height, ymax + PADDING)

                    # 坐标映射 
                    crop_box = fitz.Rect(crop_xmin, crop_ymin, crop_xmax, crop_ymax)

                    # print(f"[DEBUG] 正则匹配成功 | 描述: '{img_desc}' | 原始坐标: ({xmin}, {ymin}, {xmax}, {ymax}) | 裁剪框: {crop_box}")

                    # 裁剪并保存本地图片
                    save_filename = f"image_{image_idx}.png"
                    local_path = os.path.join(output_img_dir, save_filename)
                    
                    pix = page.get_pixmap(clip=crop_box, dpi=200)
                    pix.save(local_path)

                    abs_save_path = os.path.abspath(local_path)
                    logging.info(f"成功保存图片到: {abs_save_path}")

                    # 拼接 Markdown 输出
                    online_img_url = f"{pdf_name}/{save_filename}"
                    final_output.append(f"![IMAGE][{os.path.splitext(img_desc)[0]}]({online_img_url})")
                    image_idx += 1
                else:
                    # 兜底处理（如果正则没查到坐标）
                    print("兜底处理（如果正则没查到坐标）")
                    print(f"[Warning] 正则未匹配到的文本行内容为: {repr(line)}")
                    img_desc_match = re.search(r'!\[IMAGE\]\((.*?)\)', line)
                    img_desc = img_desc_match.group(1).strip() if img_desc_match else "图片"
                    online_img_url = f"{pdf_name}/image_{image_idx}.png"
                    final_output.append(f"![IMAGE][{img_desc}]({online_img_url})")

            

        # 最终用双换行拼接，输出格式规整、结构漂亮的最终 Markdown
        return "\n\n".join(final_output)
    
    def parse_table_special(self, table_special_str):
        """
        4. 生成最终的干净 Markdown，替换占位符为真实内容
        专门用来解析 VLM 返回的包含多个表格和坐标的长字符串
        将它们正确拆分为 [{'html': '...', 'box': '...'}, ...] 的列表
        """
        parsed_tables = []
        
        # 使用正则表达式，精准成对匹配 <table>...</table> 以及紧随其后的 <box>...</box>
        # re.DOTALL 确保 .*? 可以匹配换行符
        pattern = r'(<table>.*?</table>)\s*(<box>.*?</box>)?'
        matches = re.findall(pattern, table_special_str, re.DOTALL)
        
        for html_content, box_content in matches:
            # 清洗掉可能附带的 markdown 代码块标记
            html_clean = html_content.replace("```html", "").replace("```", "").strip()
            parsed_tables.append({
                "html": html_clean,
                "box": box_content.strip() if box_content else ""
            })
            
        logging.info(f"🛠️ 调试：从字符串中成功解析出 {len(parsed_tables)} 个表格")
        return parsed_tables
    
    def main(self,pdf_path: str = None, prompt_path: str = None, vlm = 'Qwen--Qwen3-VL-32B-Instruct'):
        if prompt_path is None:
            prompt_path = "/workspace/hf-conda/RAG/问答机器人/config/prompt_hub.yaml"

        # 先检查 YAML 文件是否存在
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"❌ 没有找到 prompt_hub.yaml 文件: {prompt_path}")

        # 加载 YAML 并检查 keys
        with open(prompt_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        prompts_dict = {p["name"]: p["content"] for p in data.get("prompts", [])}
        self.prompts = prompts_dict  # 👈 必须加这行！保证 self.prompts 不为 None

        # logging.info("📂 已加载的 prompt keys:", list(prompts_dict.keys()))
        logging.info(f"📂 已加载的 prompt keys: {list(prompts_dict.keys())}")


        # 确认必须的 prompt 是否存在
        required_keys = ["structure_prompt", "table_special_prompt_single_column"]
        for key in required_keys:
            if key not in prompts_dict:
                raise KeyError(f"❌ 缺少必要的 prompt: {key}")
        
        # 打开 PDF，获取第一页尺寸
        doc = fitz.open(pdf_path)
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
        total_pages = len(doc)
        width, height = pix.width, pix.height
        clean_doc = None
        try:
            # 1. 计算当前页面的实际像素面积
            current_pixels = width * height * total_pages
            # 2. 定义 Qwen 理想的单页最大接收 Token 数（例如：官方默认的 1280 或 16384 视显卡 VRAM 而定）
            # 这里的 1024 Token 对应大约 160 万像素 (接近 1280x1280 或者是 1600x1000)
            MAX_SINGLE_PAGE_PIXELS = 1600000
            VLM_MAX_PIXELS = MAX_SINGLE_PAGE_PIXELS
            # 3. 根据当前页面像素面积判断是否使用单页模式或滑动窗口模式
            if current_pixels <= VLM_MAX_PIXELS:
                logging.info("✅ 使用单页模式 (predict_pdf_native_single)")
                clean_doc = self.erase_template_noise(doc)
                if clean_doc is None:
                    raise ValueError(f"PDF 清洗失败或无有效内容: {pdf_path}")
                structure_prompt = self.prompts("structure_prompt")
                structure = self.predict_pdf_native_single(clean_doc, structure_prompt, vlm = vlm)
                # logging.info(f"📢 使用结构化提示词进行内容提取...{structure}")
                table_special_prompt = self.prompts("table_special_prompt_single_column")
                table_special = self.predict_pdf_native_single(clean_doc, table_special_prompt, vlm = vlm)
                final_md = self.generate_final_clean_markdown(clean_doc, structure, table_special, pdf_path)
                # ========================================================
                # 🛠️ 核心切入点：利用正则拦截并扁平化 final_md 中的所有 HTML 表格
                # ========================================================
                def flatten_all_tables_in_md(md_text):
                    # 正则匹配文本中所有的 <table>...</table> 块（忽略大小写和换行）
                    table_pattern = re.compile(r'<table[^>]*>.*?</table>', re.DOTALL | re.IGNORECASE)
                    
                    def replace_with_flat(match):
                        raw_table_html = match.group(0)
                        try:
                            # 调用你写在类下面的扁平化函数
                            # 💡 建议开启 ffill_columns="all"，防范 VLM 在复杂长表时漏标 rowspan 导致空白格
                            flat_table_html = MarkdownProcessor.process_html_to_flat_html(
                                raw_table_html
                                # ffill_columns="all" 
                            )
                            return flat_table_html if flat_table_html else raw_table_html
                        except Exception as e:
                            logging.info(f"⚠️ 单页表格扁平化失败: {e}，该表格保持原样")
                            return raw_table_html
                            
                    return table_pattern.sub(replace_with_flat, md_text)

                logging.info("📢 正在对单页最终 Markdown 中的 HTML 表格进行扁平化清洗...")
                final_md = flatten_all_tables_in_md(final_md)

                # ========================================================

                import markdown

                def validate_markdown(text: str) -> bool:
                    try:
                        html = markdown.markdown(text)
                        return bool(html.strip())  # 能成功转换成 HTML 就说明是 Markdown
                    except Exception:
                        return False
                logging.info(f"📢 验证最终生成的 Markdown 是否有效: {validate_markdown(final_md)}")

                doc.close()
                clean_doc.close()
                return final_md
            else:
                logging.info("⚙️ 使用滑动窗口模式 (structure_prompt_sliding_window)")
                clean_doc = self.erase_template_noise(doc)
                if clean_doc is None:
                    raise ValueError(f"PDF 清洗失败或无有效内容: {pdf_path}")
                scale, tile_height, overlap, rendered_w_aligned = self.compute_render_params(clean_doc)
                structure_prompt = self.prompts.get("structure_prompt_sliding_window") or self.prompts.get("structure_prompt")
                # 使用结构化提示词进行内容提取，滑动窗口模式适合大页面，但可能会有碎片化问题
                structure = self._run_vlm_stage(clean_doc, structure_prompt, tile_height=tile_height, overlap=overlap, scale=scale, rendered_w_aligned=rendered_w_aligned, vlm=vlm)
                table_special = self.extract_tables_as_html(clean_doc, structure, tile_height = tile_height  # ← 统一在这里定义
                                                            ,overlap = overlap, scale = scale,vlm = vlm)
                # print(f"📢 滑动窗口模式下，结构化内容: {structure}")
                final_md = self.generate_final_clean_markdown(clean_doc, structure, table_special, pdf_path)
                doc.close()
                clean_doc.close()
                return final_md
        finally:
        # 🌟 无论上面代码是正常 return 还是中途报错，都会雷打不动地执行这里的清理
            #  正确写法：直接利用 PyMuPDF 原生自带的布尔属性
            if doc is not None and not doc.is_closed:
                doc.close()
            if clean_doc is not None and not clean_doc.is_closed:
                clean_doc.close()
            # logging.info("💾 [内存清理] 原始 PDF 与内存清洗 PDF 对象已安全释放。")


import yaml
import os

if __name__ == "__main__":
    prompt_path = "/workspace/hf-conda/RAG/问答机器人/config/prompt_hub.yaml"
    pdf_path = '/workspace/hf-conda/RAG/问答机器人/other/finebi/函数专题/1_函数新手入门/3_运算符和优先级.pdf'

    # 初始化处理器
    processor = MarkdownProcessor(prompt_hub_path=prompt_path)
    markdown_output = processor.main(pdf_path=pdf_path, prompt_path=prompt_path, vlm="Qwen--Qwen3-VL-32B-Instruct")

    logging.info(f"============================================================================\nmarkdown_output={markdown_output}")

