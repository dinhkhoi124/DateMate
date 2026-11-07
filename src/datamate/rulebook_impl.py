"""What the executor actually implements for each scoped rule row (D18 branch 3: rule judge prompt).

Our own wording of our own code's behaviour (same paraphrase as docs/RULEBOOK_PARAMS.md), not rulebook text. The rule
judge compares this with the split's wording of the same row and decides whether the question's answer would differ.
Test: every scoped row id in rulebook.SCOPES has an entry here.
"""

IMPL: dict[str, str] = {
    "§1.1": "Đơn được tính vào doanh thu là đơn không huỷ và không trả hàng; đơn có trả hàng, kể cả trả một phần, "
            "bị bỏ cả đơn; đơn đang giao vẫn được tính.",
    "§1.2": "Hỏi bán được hay thu về bao nhiêu tiền thì trả doanh thu (theo định nghĩa đơn hợp lệ), không phải số "
            "tiền khách đã trả.",
    "§1.3": "Đơn thuộc kỳ theo ngày đặt hàng, không theo ngày giao hay ngày thu tiền.",
    "§1.4": "Đơn huỷ là đơn có trạng thái huỷ. Đơn trả hàng không được coi là đơn huỷ.",
    "§1.5": "Số đơn chỉ đếm đơn hợp lệ (không huỷ, không trả), kể cả khi đếm số đơn có một mặt hàng hay số đơn của "
            "một khách. Khách đã mua trong kỳ là khách có ít nhất một đơn hợp lệ trong kỳ.",
    "§1.6": "Doanh thu một đơn là tổng thành tiền các dòng hàng sau chiết khấu dòng; không trừ giảm giá trên cả đơn, "
            "không cộng VAT, không cộng phí giao hàng.",
    "§1.7": "Doanh thu theo hình thức thanh toán lấy theo hình thức ghi trên đơn; đơn ghi nợ không tính vào tiền mặt "
            "hay chuyển khoản; COD là hình thức riêng, không tính vào tiền mặt.",
    "§1.8": "Doanh thu theo nhóm khách lấy theo cột nhóm khách; đơn không có khách không thuộc nhóm nào; tỉ trọng "
            "nhóm chia cho doanh thu của cả shop cùng kỳ, gồm cả đơn không có khách.",
    "§1.9": "Tỉ lệ đơn huỷ = số đơn huỷ / mọi đơn đặt trong kỳ (mọi trạng thái), phần trăm làm tròn 2 chữ số.",
    "§1.10": "Phần trăm thay đổi so với kỳ trước liền kề; giảm thì ra số âm.",
    "§1.11": "Chênh lệch (số, không phải %) = kỳ hỏi trừ kỳ so, giữ đơn vị của số đo.",
    "§2.1": "Hỏi theo một ngày (ngày X, hôm nay, hôm qua) thì chỉ lấy phát sinh trong đúng ngày đó.",
    "§2.2": "Hỏi số còn lại đến hết ngày X (còn nợ, còn tồn) thì tính mọi phát sinh từ đầu dữ liệu đến hết ngày X.",
    "§2.3": "Khoảng từ X đến Y gồm cả hai đầu.",
    "§2.4": "N ngày gần nhất là N ngày liên tiếp kết thúc ở hôm nay, gồm cả hôm nay.",
    "§2.5": "Quá/hơn/trên/dưới N không gồm N; 'từ N trở lên' thì gồm N.",
    "§2.6": "Tuần bắt đầu thứ Hai. Tuần này: từ thứ Hai của tuần hiện tại đến hết hôm nay. Tuần trước: trọn tuần "
            "liền trước, thứ Hai đến Chủ nhật.",
    "§2.7": "Tháng M/Y là trọn tháng, từ ngày 1 đến ngày cuối tháng.",
    "§2.8": "Tháng này: từ ngày 1 đến hết hôm nay; so với tháng trước thì lấy cùng số ngày của tháng trước.",
    "§2.9": "Từ đầu năm: từ 01/01 năm nay đến hết hôm nay.",
    "§2.10": "Tháng hoặc quý không ghi năm thuộc năm nay nếu không sau hôm nay, sau hôm nay thì thuộc năm trước; quý "
             "theo dương lịch, 3 tháng một quý từ tháng 1.",
    "§2.11": "Tết năm Y: từ ngày dương của 23 tháng Chạp năm âm Y−1 đến hết mùng 10 tháng Giêng năm âm Y; trước Tết là "
             "cùng số ngày ngay trước 23 tháng Chạp.",
    "§2.12": "Ca sáng 06:00–13:59, ca chiều 14:00–21:59 theo giờ tạo đơn; so ca chiều với ca sáng thì mốc là ca sáng.",
    "§3.1": "Dòng 'Phí phục vụ' không phải mặt hàng: không xếp hạng, không đếm, nhưng vẫn tính vào doanh thu.",
    "§3.2": "Nhóm hàng lọc theo cột nhóm hàng, đúng tên nhóm, không gộp nhóm gần tên.",
    "§3.3": "Bán chạy xếp theo số lượng quy về đơn vị cơ bản, bỏ đơn huỷ và đơn trả; không theo số dòng hay số tiền.",
    "§3.4": "'Bán được bao nhiêu <đơn vị>' cộng số lượng quy về đơn vị cơ bản của các mặt hàng có đơn vị cơ bản đó.",
    "§3.5": "Hàng lời nhất xếp theo tổng lãi gộp của mặt hàng.",
    "§3.6": "Khách xác định theo mã khách nếu có (hai mã trùng tên là hai khách, đơn không mã không phải khách); "
            "không có mã thì theo tên, hai tên chỉ khác cách viết là một khách.",
    "§3.7": "Nhân viên bán giỏi nhất xếp theo doanh thu trong kỳ, không theo số đơn.",
    "§3.8": "Giá trị trung bình một đơn làm tròn về đồng, nửa đồng làm tròn lên.",
    "§4.1": "Công nợ xét mọi đơn trừ đơn huỷ; đơn có trả hàng vẫn được xét nếu còn tiền chưa trả.",
    "§4.2": "Nợ một đơn tại ngày X = tổng đơn (sau giảm giá cả đơn, gồm phí giao) trừ các khoản đã trả đến hết ngày X; "
            "tổng nợ cộng các đơn còn nợ dương.",
    "§4.3": "Nợ theo khách xác định khách như phần xếp hạng; đơn không có khách vẫn vào tổng nợ nhưng không có dòng "
            "trong bảng nợ theo khách.",
    "§4.4": "Đơn COD và đơn bán qua sàn không phải khách nợ, bỏ khỏi công nợ và tuổi nợ.",
    "§4.5": "Tuổi nợ tính từ ngày đặt; đặt hôm nay là 0 ngày, hôm qua là 1 ngày.",
    "§4.6": "Nợ thêm trong kỳ: phần chưa trả ngay trong ngày mua của đơn ghi nợ đặt trong kỳ. Thu nợ: các khoản trả "
            "sau ngày mua có ngày trả trong kỳ, của mọi đơn ghi nợ.",
    "§5.1": "Tiền nhập trong kỳ = tổng tiền phiếu nhập có ngày phiếu trong kỳ, bỏ phiếu huỷ; nhập đầu kỳ tính như "
            "phiếu thường. Nhà cung cấp nhập nhiều nhất theo tiền nhập.",
    "§5.2": "Nhập trừ bán: phần bán là số lượng của đơn hợp lệ, quy về cùng đơn vị với phần nhập.",
    "§5.3": "Giá nhập bình quân trong tháng tính theo số lượng nhập (tổng tiền / tổng số lượng), bỏ phiếu huỷ.",
    "§5.4": "Tồn theo đơn vị lớn trả [số đơn vị lớn nguyên, số đơn vị cơ bản lẻ].",
    "§5.5": "Giá trị tồn kho = tồn hết hôm nay của từng mặt hàng × giá vốn một đơn vị cơ bản trong danh mục hàng.",
    "§5.6": "Hết hàng cuối ngày: mặt hàng mua về để bán có tồn lúc hết ngày bằng 0.",
    "§5.7": "Hàng ế trong kỳ: đã từng bán (đơn hợp lệ) trước kỳ nhưng trong kỳ không bán được.",
    "§5.8": "Giá bán ngày D: đơn giá trên dòng bán của đơn đặt ngày D, cho một đơn vị cơ bản, trước chiết khấu dòng, "
            "gồm cả dòng của đơn huỷ và trả.",
    "§5.9": "Hiệu ứng tăng giá: ngày D giá mới cao hơn giá liền trước; so số lượng N ngày từ D (gồm D) với N ngày ngay "
            "trước D, bỏ đơn huỷ và trả.",
    "§5.10": "Giá vốn dùng giá vốn ghi trên dòng bán.",
    "§5.11": "Lãi gộp = doanh thu trừ số lượng × giá vốn ghi trên dòng bán (cho một đơn vị bán).",
    "§5.12": "Lãi ròng tháng = lãi gộp của tháng trừ mọi khoản trong sổ chi phí của tháng đó.",
    "§6[ca phe].1": "Lọc size M/L theo hậu tố tên món là (M) hay (L), so khớp bỏ qua chữ hoa/thường.",
    "§6[nha thuoc].1": "Thuốc kê đơn là sản phẩm ghi kê đơn trong danh mục thuốc; tỉ lệ kê đơn tính theo doanh thu.",
    "§6[nha thuoc].2": "Hết hạn trong N ngày tới: lô còn tồn có hạn từ hôm nay đến hết hôm nay + N; tiền vốn là giá trị "
                       "phần còn tồn theo giá nhập của lô.",
    "§6[nha thuoc].3": "Doanh thu theo hoạt chất: thuốc có hoạt chất ghi đúng như vậy; thuốc phối hợp là loại riêng.",
    "§6[nha thuoc].4": "Lô gần hết hạn nhất: trong các lô còn tồn chưa quá hạn, lô có hạn sớm nhất.",
    "§6[vat lieu xay dung].1": "Số đợt giao trong tháng: mỗi đợt của mỗi đơn có ngày giao trong tháng tính một đợt.",
    "§10.1": "Số lượng quy về đơn vị cơ bản theo bảng quy đổi của shop; đơn vị không có dòng quy đổi có hệ số 1.",
}
