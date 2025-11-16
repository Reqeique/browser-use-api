# processes
ps aux | egrep 'chrom(e|ium)|playwright|puppeteer|chromedriver'

# open fds & locks
lsof -nP | egrep 'chrome|chromium'

# recent kernel / selinux / apparmor messages
dmesg | tail -n 50
sudo journalctl -u your_service_name -n 200 --no-pager

# free/disk
free -m
df -h

# try manual launch (example)
/path/to/chrome --headless --disable-gpu --remote-debugging-port=9222 --enable-logging --v=1
