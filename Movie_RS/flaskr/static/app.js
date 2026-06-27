// Vue app for the Enhanced UI.
// Handles: profile switcher, genre picker, movie search, star rating,
//          "haven't seen" interest flags, like/dislike, diversity slider,
//          detail modal, explanation popover, feedback diary, Your Taste
//          dashboard.
import {createApp, ref, computed, onMounted} from
    'https://unpkg.com/vue@3/dist/vue.esm-browser.js'

const DATA = window.__SERVER_DATA__ || {};
const PROFILE_IDS = DATA.profileIds || ['P01', 'P02', 'P03', 'P04', 'P05', 'P06'];
const PROFILE_ID = DATA.profileId || '';

function _prefixed(name) {
    return PROFILE_ID ? `${PROFILE_ID}_${name}` : name;
}
function setCookie(name, value, days = 30, prefixed = true) {
    const d = new Date();
    d.setTime(d.getTime() + days * 24 * 3600 * 1000);
    const key = prefixed ? _prefixed(name) : name;
    document.cookie = `${key}=${value};expires=${d.toUTCString()};path=/`;
}
function clearCookie(name, prefixed = true) {
    const key = prefixed ? _prefixed(name) : name;
    document.cookie = `${key}=; Max-Age=0; path=/`;
}
function getCookie(name, prefixed = true) {
    const key = prefixed ? _prefixed(name) : name;
    const m = document.cookie.match(new RegExp('(^| )' + key + '=([^;]+)'));
    return m ? m[2] : null;
}

const app = createApp({
    setup() {
        const isLoading = ref(true)
        const showGenres = ref(false)
        const showDiary = ref(false)
        const showTaste = ref(false)
        const detailMovie = ref(null)
        const detailExplanation = ref(null)
        const explainMovie = ref(null)
        const interestMovie = ref(null)

        const profileId = ref(PROFILE_ID)
        const profileIds = ref(PROFILE_IDS)
        const genres = ref(DATA.userGenres || [])
        const userRates = ref(DATA.userRates || [])
        const userLikes = ref(DATA.userLikes || [])
        const userDislikes = ref(DATA.userDislikes || [])
        const userInterests = ref(DATA.userInterests || [])
        const explanations = ref(DATA.explanations || {})
        const diversity = ref(Math.round(parseFloat(getCookie('diversity') || '0.25') * 100))

        const searchQuery = ref('')
        const searchResults = ref([])

        const tasteStats = ref(null)

        // Onboarding progress: a movie the user has rated OR flagged as
        // interested/not-interested/skipped counts as "handled", so the
        // 10-movie gate advances even when the user clicks "haven't seen".
        const onboardingProgress = computed(() => {
            const seen = new Set()
            userRates.value.forEach(e => {
                const p = e.split('|')
                if (p.length >= 2) seen.add(p[1])
            })
            userInterests.value.forEach(e => {
                const p = e.split('|')
                if (p.length >= 2) seen.add(p[0])
            })
            return seen.size
        })

        onMounted(() => {
            isLoading.value = false
            if (genres.value.length === 0) showGenres.value = true
        })

        // --- Genres ---
        function updateGenre(id) {
            const s = id.toString()
            const i = genres.value.indexOf(s)
            if (i > -1) genres.value.splice(i, 1)
            else {
                genres.value.push(s)
                genres.value.sort((a, b) => a - b)
            }
        }
        const inGenres = id => genres.value.includes(id.toString())
        function saveGenres() {
            setCookie('user_genres', genres.value.join(','))
            clearCookie('user_rates')
            reloadPage()
        }

        // --- Ratings ---
        function getRate(movieId) {
            for (let i in userRates.value) {
                const r = userRates.value[i].split('|')
                if (parseInt(r[1]) === movieId) return [i, parseInt(r[2])]
            }
            return [-1, -1]
        }
        function updateRate(movieId, rate) {
            const ts = Math.floor(Date.now() / 1000)
            const record = `611|${movieId}|${rate}|${ts}`
            const i = getRate(movieId)[0]
            if (i > -1) userRates.value[i] = record
            else userRates.value.push(record)
            setCookie('user_rates', userRates.value.join(','))
        }

        // --- Likes / Dislikes ---
        function updateLike(movieId) {
            const s = movieId.toString()
            const i = userLikes.value.indexOf(s)
            if (i > -1) userLikes.value.splice(i, 1)
            else {
                userLikes.value.push(s)
                const j = userDislikes.value.indexOf(s)
                if (j > -1) userDislikes.value.splice(j, 1)
            }
            setCookie('user_likes', userLikes.value.join(','))
            setCookie('user_dislikes', userDislikes.value.join(','))
            reloadPage()
        }
        function updateDislike(movieId) {
            const s = movieId.toString()
            const i = userDislikes.value.indexOf(s)
            if (i > -1) userDislikes.value.splice(i, 1)
            else {
                userDislikes.value.push(s)
                const j = userLikes.value.indexOf(s)
                if (j > -1) userLikes.value.splice(j, 1)
            }
            setCookie('user_likes', userLikes.value.join(','))
            setCookie('user_dislikes', userDislikes.value.join(','))
            reloadPage()
        }
        const inLikes = id => userLikes.value.includes(id.toString())
        const inDislikes = id => userDislikes.value.includes(id.toString())

        // --- "Haven't seen" interest flags ---
        function _interestIndex(movieId) {
            const s = movieId.toString()
            return userInterests.value.findIndex(e => e.split('|')[0] === s)
        }
        function interestOf(movieId) {
            const i = _interestIndex(movieId)
            if (i < 0) return null
            return userInterests.value[i].split('|')[1] || null
        }
        async function openInterest(movieId) {
            const r = await fetch('/api/movie/' + movieId)
            if (!r.ok) return
            interestMovie.value = await r.json()
        }
        function setInterest(movieId, status) {
            const s = movieId.toString()
            const i = _interestIndex(movieId)
            const record = `${s}|${status}`
            if (i > -1) userInterests.value[i] = record
            else userInterests.value.push(record)
            setCookie('user_interests', userInterests.value.join(','))
            interestMovie.value = null
            reloadPage()
        }

        // --- Search ---
        let searchTimer = null
        function doSearch() {
            clearTimeout(searchTimer)
            searchTimer = setTimeout(async () => {
                const q = searchQuery.value.trim()
                if (q.length < 2) { searchResults.value = []; return }
                const r = await fetch('/api/search?q=' + encodeURIComponent(q))
                searchResults.value = await r.json()
            }, 250)
        }

        // --- Detail / explain modals ---
        async function openDetail(movieId) {
            const r = await fetch('/api/movie/' + movieId)
            if (!r.ok) return
            detailMovie.value = await r.json()
            const er = await fetch('/api/explain/' + movieId)
            detailExplanation.value = er.ok ? await er.json() : null
        }
        async function openExplain(movieId) {
            const movie = await (await fetch('/api/movie/' + movieId)).json()
            const expl = await (await fetch('/api/explain/' + movieId)).json()
            explainMovie.value = Object.assign({}, movie, expl)
        }

        // --- Diversity slider ---
        function onDiversityChange(e) {
            diversity.value = parseInt(e.target.value)
            setCookie('diversity', (diversity.value / 100).toFixed(2))
            // Debounce auto-reload to avoid flooding server while dragging.
            clearTimeout(window.__divTimer)
            window.__divTimer = setTimeout(reloadPage, 600)
        }

        // --- Feedback diary titles ---
        const likedTitles = ref([])
        const dislikedTitles = ref([])
        const interestTitles = ref([])
        async function refreshDiary() {
            likedTitles.value = await Promise.all(
                userLikes.value.map(async id => (await (await fetch('/api/movie/' + id)).json()).title)
            )
            dislikedTitles.value = await Promise.all(
                userDislikes.value.map(async id => (await (await fetch('/api/movie/' + id)).json()).title)
            )
            const pairs = userInterests.value
                .map(e => e.split('|'))
                .filter(p => p.length === 2)
            interestTitles.value = await Promise.all(pairs.map(async ([id, st]) => {
                const m = await (await fetch('/api/movie/' + id)).json()
                return {id, title: m.title || ('#' + id), status: st}
            }))
        }
        refreshDiary()

        // --- Profile switcher ---
        function switchProfile(pid) {
            if (!pid || pid === profileId.value) return
            setCookie('profile_id', pid, 30, false)
            reloadPage()
        }
        function cleanAll() {
            if (!confirm('Delete the data for profile ' +
                (profileId.value || '(default)') + '?')) return
            ;['user_genres', 'user_rates', 'user_likes', 'user_dislikes',
              'user_interests', 'diversity'].forEach(n => clearCookie(n))
            reloadPage()
        }

        // --- Your Taste dashboard ---
        async function openTaste() {
            const r = await fetch('/api/profile/stats')
            tasteStats.value = r.ok ? await r.json() : null
            showTaste.value = true
            // Let the modal render before we draw into canvases.
            setTimeout(renderTasteCharts, 50)
        }
        function renderTasteCharts() {
            if (!tasteStats.value || typeof Chart === 'undefined') return
            const s = tasteStats.value
            const radarEl = document.getElementById('tasteRadar')
            if (radarEl) {
                if (window.__tasteRadar) window.__tasteRadar.destroy()
                const labels = Object.keys(s.genre_ratings || {})
                const values = labels.map(l => s.genre_ratings[l])
                window.__tasteRadar = new Chart(radarEl, {
                    type: 'radar',
                    data: {labels, datasets: [{
                        label: 'Avg score',
                        data: values,
                        backgroundColor: 'rgba(13,110,253,.2)',
                        borderColor: 'rgba(13,110,253,.9)',
                    }]},
                    options: {scales: {r: {min: 0, max: 5}}},
                })
            }
            const barEl = document.getElementById('tasteDecade')
            if (barEl) {
                if (window.__tasteDecade) window.__tasteDecade.destroy()
                const d = s.decade_counts || {}
                const labels = Object.keys(d).sort()
                const values = labels.map(l => d[l])
                window.__tasteDecade = new Chart(barEl, {
                    type: 'bar',
                    data: {labels, datasets: [{
                        label: 'Movies',
                        data: values,
                        backgroundColor: 'rgba(25,135,84,.7)',
                    }]},
                    options: {plugins: {legend: {display: false}}},
                })
            }
        }

        const reloadPage = () => location.reload()

        return {
            isLoading, showGenres, showDiary, showTaste,
            detailMovie, detailExplanation, explainMovie, interestMovie,
            profileId, profileIds,
            genres, userRates, userLikes, userDislikes, userInterests,
            explanations, diversity,
            searchQuery, searchResults,
            likedTitles, dislikedTitles, interestTitles,
            tasteStats, onboardingProgress,
            updateGenre, inGenres, saveGenres,
            getRate, updateRate,
            updateLike, updateDislike, inLikes, inDislikes,
            openInterest, setInterest, interestOf,
            doSearch, openDetail, openExplain,
            onDiversityChange, cleanAll, reloadPage,
            switchProfile, openTaste,
        }
    }
})
app.config.compilerOptions.delimiters = ['[[', ']]']
app.mount('#app')
