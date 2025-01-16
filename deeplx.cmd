@echo off
curl -X POST http://localhost:1188/translate ^
-H "Content-Type: application/json" ^
-H "Authorization: Bearer your_access_token" ^
-d "{\"text\": \"Hello, world!\", \"source_lang\": \"EN\", \"target_lang\": \"DE\"}" 