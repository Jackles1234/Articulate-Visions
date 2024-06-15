module.exports = {
    async rewrites() {
        return [
            {
                source: '/api/:path*',
                destination: 'http://3.236.122.207:5000/:path*',
            },
        ]
    },
};