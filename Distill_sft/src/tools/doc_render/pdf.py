from typing import List

import fitz


def pdf_to_images(pdf_data) -> List[bytes]:
    """
    将PDF文件的每一页转换为 PNG 图像格式，并返回包含所有页图像数据的字节列表。

    Args:
        pdf_data (bytes): PDF文件的字节流数据。

    Returns:
        List[bytes]: 每个元素为 PDF 一页的 PNG 图像字节数据。
    """
    pdf_document = fitz.open(stream=pdf_data)
    images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        data = pix.tobytes("png")
        images.append(data)
    return images


