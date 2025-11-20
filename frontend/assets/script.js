async function getRecommendations() {
    const input = document.getElementById('movieInput');
    const resultsDiv = document.getElementById('results');
    const movieName = input.value;

    if (!movieName) return alert("Please enter a movie name!");
    resultsDiv.innerHTML = '<p>Thinking...</p>';
    try {
        const response = await fetch(`http://127.0.0.1:8000/recommend/${movieName}`);
        if (!response.ok) throw new Error("Movie not found! Try the exact English title.");
        const data = await response.json();
        resultsDiv.innerHTML = ''; 
        data.recommendations.forEach(movie => {
            const card = document.createElement('div');
            card.className = 'card';
            card.innerHTML = `<h3>${movie}</h3>`;
            resultsDiv.appendChild(card);
        });

    } catch (error) {
        resultsDiv.innerHTML = `<p style="color: orange;">${error.message}</p>`;
    }
}