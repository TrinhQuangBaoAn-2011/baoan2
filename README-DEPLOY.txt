CHỈ UPLOAD FILE NÀY LÊN VERCEL:

UPLOAD-THIS-vaio-website-static.zip
(hoặc vaio-website-static.zip — cùng nội dung)

KHÔNG upload:
- vaio-website-vercel.zip
- thư mục vaio-website-vercel
- các file vaio-website-static-*.zip cũ
- bất kỳ thư mục source nào

Sau khi deploy đúng, kiểm tra:
1. View Source của web phải thấy /assets/index-CkE3sGvL.js
2. Vào Cài đặt, dòng "Bản build" phải là 22 thg 10, 2026 21:55
3. Trang đăng nhập phải hiện nút "Đăng nhập bằng Google"
   (không còn ô nhập email local)

Nếu vẫn thấy ô nhập email local:
- Đã upload nhầm bản source hoặc bản build không có .env.local.
- Upload lại đúng file zip ở trên.
