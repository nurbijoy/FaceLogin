const user = JSON.parse(localStorage.getItem('user'));

if (!user) {
    window.location.href = '/';
} else {
    document.getElementById('username').textContent = user.username;
    document.getElementById('email').textContent = user.email;
    document.getElementById('username2').textContent = user.username;
    document.getElementById('email2').textContent = user.email;
    document.getElementById('loginTime').textContent = new Date().toLocaleString();
}

document.getElementById('logoutBtn').addEventListener('click', () => {
    localStorage.removeItem('user');
    window.location.href = '/';
});
