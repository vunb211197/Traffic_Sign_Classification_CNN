# Traffic_Sign_Classification_CNN
Bài toán phân loại biển báo giao thông trên đường sử dụng mô hình CNN

# Giới thiệu về data của bài toán Trafic Sign Recognition Bench Mark
 - Data này bao gồm các dạng biển báo giao thông của Đức
 - Tập train được sử dụng cho cuộc thi trực tuyến IJCNN2011 và đã được public sau khi cuộc thi kết thúc
 - Data gồm có 43 thư mục tương ứng với 43 class,mỗi thư mục chứa các hình ảnh train tương ứng và một file csv để chú thích 
 - Định dạng hình ảnh và đặt tên :
    - Hình ảnh được về dạng PPM(màu RGB).Các tệp được đánh số thành hai phần :
        XXXXX_YYYYY.ppm
    - Nói qua về định dạng PPM : 
    File ppm là phần mở rộng của file ảnh 24-bit màu được định dạng bằng định dạng văn bản, chứa chiều rộng và chiều cao của ảnh, giá trị màu tuyệt đối và các dữ liệu khoảng trắng.
    - Phần đầu tiên , XXXXX đại diện cho số theo dõi .Tất cả hình ảnh của một lớp với các số theo dõi giống nhau bắt nguồn từ một biển báo giao thông thực tế 
    - Phần thứ 2 ,YYYYY là một con số đang chạy trong theo dõi .Trật tự thời gian trong ảnh được giữ nguyên 
- Định dạng phần chú thích (file csv)
    - Chú thích chứa thông tin về hình ảnh và class ID
    - Cụ thể , nó cung cấp các trường sau :
        - File_name : tên tệp  hình ảnh
        - Width,Height : kích thước chiều rộng và chiều cao của hình ảnh 
        - Roi.x1, Roi.y1,Roi.x2, Roi.y2 : vị trí của dấu hiệu trong hình ảnh (hình ảnh có đường viền bao quanh biển báo thực tế )
        - Class ID : Class của biển báo đấy là gì 
