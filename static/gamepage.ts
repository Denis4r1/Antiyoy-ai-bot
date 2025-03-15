// ===================== ПАРСИНГ room_id token username из URL =====================
let roomId: string | null = null;
let username: string | null = null;
let token: string | null = null;

{
    const urlParams = new URLSearchParams(window.location.search);
    roomId = urlParams.get("room_id");
    username = urlParams.get("username");
    token = urlParams.get("token");
    if (!roomId) {
        roomId = null;
    }
    if (!username) {
        username = null;
    }
    if (!token) {
        token = null;
    }
}

const userNameDisplayElem = document.getElementById("userNameDisplay");
if (userNameDisplayElem) {
    userNameDisplayElem.textContent = `Ваше имя: ${username}`;
}

// ===================== WebSocket-подключение =====================
const scheme = window.location.protocol === "https:" ? "wss" : "ws";
const wsUrl = `${scheme}://${window.location.host}/ws/game/${roomId}/${token}/${username}`;

let ws: WebSocket | null = new WebSocket(wsUrl);

/**
 * Получаем сообщения от сервера по WebSocket.
 */
if (ws) {
    ws.onopen = () => {
        console.log("WebSocket connected:", wsUrl);
    };

    ws.onmessage = (event: MessageEvent) => {
        try {
            const data = JSON.parse(event.data);
            switch (data.type) {
                case "game_state_update":
                    gameState = {
                        rows: data.field.height,
                        cols: data.field.width,
                        current_turn: data.current_player,
                        players: data.players, // если нужно
                        field: data.field
                    };
                    // Обновляем canvas
                    canvas.width = gameState.cols * 1.5 * radius + radius;
                    canvas.height = gameState.rows * Math.sqrt(3) * radius + radius;
                    selectedCellCoord = null;
                    availableMoves = [];
                    drawMap();
                    updateGameInfo();
                    break;

                case "available_moves":
                    
                    availableMoves = data.moves;
                    drawMap();
                    break;

                case "game_over":
                    alert(`winner ${data.winner}`);
                    window.location.href = `/?room_id=${data.lobby_id}&username=${username}&token=${token}`;
                    break;
                    break;
                
                case "error":
                    showStatus(data.message);

                default:
                    console.log("WS message:", data);
                    break;
            }
        } catch {
            // Если не JSON, просто логируем
            console.log("WS message (raw):", event.data);
        }
    };

    ws.onclose = () => {
        console.log("WebSocket closed.");
        ws = null;
    };

    ws.onerror = (err) => {
        console.error("WebSocket error:", err);
    };
}

// ===================== ОСНОВНЫЕ ПЕРЕМЕННЫЕ И НАСТРОЙКИ ИГРЫ =====================
interface GameState {
    rows: number;
    cols: number;
    current_turn: string;
    players: Array<{
        name: string;
        coins: number;
    }>;
    field: {
        cells: {
            [key: string]: {
                owner?: string;
                entity?: string; // "unit{n}", "farm", "empty", "base",...
            };
        };
        territories?: Array<{
            [playerName: string]: string[];
        }>;
    };
}

let gameState: GameState | null = null;


const canvas = document.getElementById('gameCanvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d');

const radius = 30;    // Радиус шестиугольника
const iconSize = 40;  // Размер иконки

interface CellCoord {
    x: number; // j
    y: number; // i
}

let selectedCellCoord: CellCoord | null = null;   // Выбранная клетка (если есть)
let hoveredCell: CellCoord | null = null;         // Клетка под курсором
let availableMoves: CellCoord[] = [];             // Доступные ходы для выбранного юнита


const icons: Record<string, HTMLImageElement> = {
    "unit1": new Image(),
    "unit2": new Image(),
    "unit3": new Image(),
    "unit4": new Image(),
    "base": new Image(),
    "farm": new Image(),
    "weakTower": new Image(),
    "strongTower": new Image()
};
icons.unit1.src = "/static/unit1.png";
icons.unit2.src = "/static/unit2.png";
icons.unit3.src = "/static/unit3.png";
icons.unit4.src = "/static/unit4.png";
icons.base.src = "/static/base.png";
icons.farm.src = "/static/farm.png";
icons.weakTower.src = "/static/weakTower.png";
icons.strongTower.src = "/static/strongTower.png";


async function loadAllIcons(): Promise<void> {
    const images = Object.values(icons);

    const promises = images.map(img => {
        return new Promise<void>((resolve, reject) => {
            if (img.complete) {
                resolve();
            } else {
                img.onload = () => resolve();
                img.onerror = () => reject(new Error("Не удалось загрузить " + img.src));
            }
        });
    });

    // Ждём загрузки всех иконок
    await Promise.all(promises);
    // После того как все иконки загружены, перерисовываем карту (До этого вместо иконок был написан текст)
    drawMap();
}


const localPlayerColor = "#B2E4B2";
const pastelColorPool = ["#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF", "#D3BAFF", "#FFBAF0"];



const colors: Record<string, string> = {
    null: '#EEEEEE',
};


/**
 * Генерация случайного пастельного цвета (если вдруг закончится пул).
 */
function generatePastelColor(): string {
    const r = Math.floor(Math.random() * 127 + 127);
    const g = Math.floor(Math.random() * 127 + 127);
    const b = Math.floor(Math.random() * 127 + 127);
    const hr = r.toString(16).padStart(2, '0');
    const hg = g.toString(16).padStart(2, '0');
    const hb = b.toString(16).padStart(2, '0');
    return `#${hr}${hg}${hb}`;
}

/**
 * Показать сообщение в блоке #status на 2 секунды. (ошибки, неправильные ходы)
 */
function showStatus(message: string) {
    const statusElem = document.getElementById('status');
    if (!statusElem) return;
    statusElem.textContent = message;
    setTimeout(() => {
        if (statusElem.textContent === message) {
            statusElem.textContent = '';
        }
    }, 2000);
}


// ===================== DRAWING HEX MAP =====================
function drawHex(x: number, y: number, color: string, text: string = '') {
    if (!ctx) return;
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
        const angle = (Math.PI / 3) * i;
        const hx = x + radius * Math.cos(angle);
        const hy = y + radius * Math.sin(angle);
        i === 0 ? ctx.moveTo(hx, hy) : ctx.lineTo(hx, hy);
    }
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = '#000000';
    ctx.stroke();
    if (text) {
        ctx.fillStyle = '#000000';
        ctx.fillText(text, x - 5, y + 5);
    }
}

function drawMap() {
    if (!ctx || !gameState || !gameState.field || !gameState.field.cells) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    Object.entries(gameState.field.cells).forEach(([key, cell]) => {
        const [iStr, jStr] = key.split(',');

        const i = parseInt(iStr, 10);
        const j = parseInt(jStr, 10);

        const x = j * 1.5 * radius + radius;
        let y = i * Math.sqrt(3) * radius + radius;
        if (j % 2 === 1) {
            y += (Math.sqrt(3) * radius) / 2;
        }

        // Определяем цвет 
        let ownerColor = colors.null;
        if (cell.owner) {
            if (cell.owner === username) {
                ownerColor = localPlayerColor;
            } else {
                if (!colors[cell.owner]) {
                    const assignedColors = new Set(Object.values(colors));
                    assignedColors.delete(localPlayerColor);
                    
                    const availableColors = pastelColorPool.filter(color => !assignedColors.has(color));
                    if (availableColors.length > 0) {
                    
                        colors[cell.owner] = availableColors[0];
                    } else {
                        colors[cell.owner] = generatePastelColor();
                    }
                }
                ownerColor = colors[cell.owner];
            }

        }

        // Рисуем шестиугольник
        drawHex(x, y, ownerColor);

        // Рисуем иконку / текст сущности (если не загрузилось)
        if (cell.entity && cell.entity !== "empty") {
            const icon = icons[cell.entity];
            if (icon && icon.complete) {
                ctx.drawImage(icon, x - iconSize / 2, y - iconSize / 2, iconSize, iconSize);
            } else {
                ctx.fillStyle = "#000000";
                ctx.fillText(cell.entity, x - 10, y + 5);
            }
        }

        // Подсветка выбранной клетки
        if (selectedCellCoord && selectedCellCoord.x === j && selectedCellCoord.y === i) {
            ctx.strokeStyle = '#00FF00';
            ctx.lineWidth = 3;
            ctx.beginPath();
            for (let k = 0; k < 6; k++) {
                const angle = (Math.PI / 3) * k;
                const hx = x + radius * Math.cos(angle);
                const hy = y + radius * Math.sin(angle);
                k === 0 ? ctx.moveTo(hx, hy) : ctx.lineTo(hx, hy);
            }
            ctx.closePath();
            ctx.stroke();
            ctx.lineWidth = 1;
        }

        // Подсветка клетки под курсором 
        if (hoveredCell && hoveredCell.x === j && hoveredCell.y === i) {
            ctx.strokeStyle = '#FFFF00';
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (let k = 0; k < 6; k++) {
                const angle = (Math.PI / 3) * k;
                const hx = x + radius * Math.cos(angle);
                const hy = y + radius * Math.sin(angle);
                k === 0 ? ctx.moveTo(hx, hy) : ctx.lineTo(hx, hy);
            }
            ctx.closePath();
            ctx.stroke();
            ctx.lineWidth = 1;

            ctx.fillStyle = "#000000";
            ctx.font = "14px Arial";
            ctx.fillText(`(${i},${j})`, x - 15, y + 5);
        }

        // Подсветка доступных ходов
        if (availableMoves.some(m => m.x === j && m.y === i)) {
            ctx.strokeStyle = '#0000FF';
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (let k = 0; k < 6; k++) {
                const angle = (Math.PI / 3) * k;
                const hx = x + radius * Math.cos(angle);
                const hy = y + radius * Math.sin(angle);
                k === 0 ? ctx.moveTo(hx, hy) : ctx.lineTo(hx, hy);
            }
            ctx.closePath();
            ctx.stroke();
            ctx.lineWidth = 1;
        }
    });
}



// Основная функция для общения с сервером
function sendGameAction(type: string, payload: any = {}) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        showStatus("WebSocket не подключен!");
        return;
    }
    ws.send(JSON.stringify({ type, payload }));
}


// ===================== END_TURN =====================
async function endTurn() {
    sendGameAction("end_turn");
}

// ===================== BUILD =====================
async function build(building: string) {
    if (!roomId || !username) return;
    if (!selectedCellCoord) {
        showStatus("Select a cell first!");
        return;
    }
    sendGameAction("build_action", {
        x: selectedCellCoord.x,
        y: selectedCellCoord.y,
        building: building
    });

    selectedCellCoord = null;
    availableMoves = [];
}

// ===================== SPAWN UNIT =====================
async function spawnUnit(level: number) {
    if (!roomId || !username) return;
    if (!selectedCellCoord) {
        showStatus("Select a cell first!");
        return;
    }
    sendGameAction("spawn_unit", {
        x: selectedCellCoord.x,
        y: selectedCellCoord.y,
        level: level
    });

    selectedCellCoord = null;
    availableMoves = [];
}

// ===================== GET_MOVES =====================
async function getMoves(x: number, y: number) {
    if (!roomId || !username) return;
    sendGameAction("get_moves", { x, y });
}




function deselectCell() {
    selectedCellCoord = null;
    availableMoves = [];
    drawMap();
}

// Вычислить координаты шестиугольной ячейки по координатам на экране
function getHexCoordinates(e: MouseEvent): CellCoord | null {
    if (!gameState) return null;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const j = Math.floor(mouseX / (1.5 * radius));
    let yCoord = mouseY;
    if (j % 2 === 1) {
        yCoord -= (Math.sqrt(3) * radius) / 2;
    }
    const i = Math.floor(yCoord / (Math.sqrt(3) * radius));

    // проверка границ
    if (i >= 0 && i < gameState.rows && j >= 0 && j < gameState.cols) {
        return { x: j, y: i };
    }
    return null;
}

// Подсветка клетки под курсором
canvas.addEventListener('mousemove', (e) => {
    hoveredCell = getHexCoordinates(e);
    drawMap();
});

// Убираем подсветку при уходе курсора
canvas.addEventListener('mouseleave', () => {
    hoveredCell = null;
    drawMap();
});

canvas.addEventListener('click', async (e) => {
    if (!gameState) return;
    const coord = getHexCoordinates(e);
    if (!coord) return;

    const key = `${coord.y},${coord.x}`;
    const cell = gameState.field.cells[key];
    if (!cell) return;

    // Если уже что-то выбрано, значит пытаемся сделать move
    if (selectedCellCoord) {
        if (selectedCellCoord.x === coord.x && selectedCellCoord.y === coord.y) {
            // Кликнули по той же клетке = снимаем выбор
            deselectCell();
            return;
        }

        // Если нечего двигать (нет было юнитов в предыдущей выбранной клетке) то просто выбираем клетку
        // Но если в новой оказался юнит, то сделаем то же самое, что и при простом клике на клетку с юнитом
        // # TODO clean up 
        const selectedCellKey = `${selectedCellCoord.y},${selectedCellCoord.x}`
        const selectedCell = gameState.field.cells[selectedCellKey];
        if (selectedCell.entity && !selectedCell.entity.startsWith("unit") || selectedCell.owner != gameState.current_turn) {
            if (cell.entity && cell.entity.startsWith("unit") && cell.owner === gameState.current_turn) {
                // Кликнули на своего юнита
                selectedCellCoord = { x: coord.x, y: coord.y };
                await getMoves(coord.x, coord.y);
            } else {
                selectedCellCoord = { x: coord.x, y: coord.y };
                availableMoves = [];
                drawMap();
            }
            return;
        }
        // Иначе пробуем переместить юнита
        try {
            const response = sendGameAction("move_unit", {
                from_x: selectedCellCoord.x,
                from_y: selectedCellCoord.y,
                to_x: coord.x,
                to_y: coord.y
            });

            selectedCellCoord = null;
            availableMoves = [];
        } catch (err: any) {
            showStatus(`Error: ${err.message}`);
        }
    } else {
        // Не было выбранных клеток
        // Если тут есть "unit" и владелец == текущему игроку, тогда покажем куда юнит может походить
        if (cell.entity && cell.entity.startsWith("unit") && cell.owner === gameState.current_turn) {
            selectedCellCoord = { x: coord.x, y: coord.y };
            await getMoves(coord.x, coord.y);
        } else {
            selectedCellCoord = { x: coord.x, y: coord.y };
            availableMoves = [];
            drawMap();
        }
    }
});



// ========== ПРИВЯЗКА КНОПОК ==========
const endTurnBtn = document.getElementById("endTurnBtn");
endTurnBtn?.addEventListener("click", () => {
    sendGameAction("end_turn");
});

const buildFarmBtn = document.getElementById("buildFarmBtn");
buildFarmBtn?.addEventListener("click", () => {
    build("farm");
});

const buildWeakTowerBtn = document.getElementById("buildWeakTowerBtn");
buildWeakTowerBtn?.addEventListener("click", () => {
    build('weakTower');
});

const buildStrongTowerBtn = document.getElementById("buildStrongTowerBtn");
buildStrongTowerBtn?.addEventListener("click", () => {
    build("strongTower");
});

const spawnUnit1Btn = document.getElementById("spawnUnit1Btn");
spawnUnit1Btn?.addEventListener("click", () => {
    spawnUnit(1);
});

const spawnUnit2Btn = document.getElementById("spawnUnit2Btn");
spawnUnit2Btn?.addEventListener("click", () => {
    spawnUnit(2);
});

const spawnUnit3Btn = document.getElementById("spawnUnit3Btn");
spawnUnit3Btn?.addEventListener("click", () => {
    spawnUnit(3);
});

const spawnUnit4Btn = document.getElementById("spawnUnit4Btn");
spawnUnit4Btn?.addEventListener("click", () => {
    spawnUnit(4);
});

const deselectBtn = document.getElementById("deselectBtn");
deselectBtn?.addEventListener("click", () => {
    deselectCell();
});


function updateGameInfo() {
    const statusDiv = document.getElementById("turnStatus");
    if (!gameState) return;
    statusDiv.innerHTML = `Current Turn: ${gameState.current_turn} <br>` +
        (gameState.current_turn ===  username?
            `<span style="color: green;">Ваш ход!</span>` :
            `<span style="color: red;">Ход противника!</span>`);

    const territoryDiv = document.getElementById('territoryInfo');
    if (!territoryDiv) return;
    let territoryHTML = '<h3>Territories</h3>';
    if (gameState.field && gameState.field.territories && gameState.field.territories.length > 0) {
        gameState.field.territories.forEach((entry) => {
            for (const player in entry) {
                territoryHTML += `<strong>${player}:</strong><br>`;
                entry[player].forEach(infoStr => {
                    territoryHTML += `${infoStr}<br>`;
                });
            }
        });
    } else {
        territoryHTML += 'No territories';
    }
    territoryDiv.innerHTML = territoryHTML;
}



loadAllIcons()
    .catch((error) => {
        console.error('Ошибка при загрузке иконок:', error);
    });
