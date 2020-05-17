import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.staticfiles import StaticFiles
import base64
from PIL import Image
import matplotlib.pyplot as plt

codes = ['Animal', 'Archway', 'Bicyclist', 'Bridge', 'Building', 'Car', 'CartLuggagePram', 'Child', 'Column_Pole',
       'Fence', 'LaneMkgsDriv', 'LaneMkgsNonDriv', 'Misc_Text', 'MotorcycleScooter', 'OtherMoving', 'ParkingBlock',
       'Pedestrian', 'Road', 'RoadShoulder', 'Sidewalk', 'SignSymbol', 'Sky', 'SUVPickupTruck', 'TrafficCone',
       'TrafficLight', 'Train', 'Tree', 'Truck_Bus', 'Tunnel', 'VegetationMisc', 'Void', 'Wall']

def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

export_file_url = 'https://www.dropbox.com/s/dth7kg6dak6x3sd/export.pkl?dl=1'
export_file_name = 'export.pkl'

classes = ['black', 'grizzly', 'teddys']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    img = img.resize(416)
    outputs = learn.predict(img)
    cm_hot = plt.get_cmap('tab10')
    masked = outputs[0].data
    im = np.array(masked)
    im2 = np.squeeze(im)
    im_color = cm_hot(im2)
    resp_bytes = BytesIO()
    PIL.Image.fromarray((im_color*255).astype('uint8')).save(resp_bytes, format='png')
    background=Image.open(BytesIO(img_bytes))
    background=background.resize((416,416))
    overlay=Image.open(resp_bytes)
    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")
    new_img = Image.blend(background, overlay, 0.5)
    buffered = BytesIO()
    new_img.save(buffered, format="png")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    img_str = "data:image/png;base64," + img_str
    return JSONResponse({'result': str(img_str)})



if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")

